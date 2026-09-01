"""KV restoration orchestration and selectable INT8 dequantization strategies."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from .arena import ArenaKVBlock
from .kv_types import LayeredStoredBlock, LegacyKV, StoredBlock
from .tensors import QuantizedTensor, concatenate_caches


INT8_RESTORE_BACKENDS = ("pytorch", "auto", "triton")
DequantizePair = Callable[..., tuple[object, object]]


@dataclass
class RestoreResult:
    cache: LegacyKV
    backend: str
    load_s: float = 0.0
    transfer_s: float = 0.0
    dequant_s: float = 0.0


def resolve_int8_restore_backend(requested: str, device) -> str:
    if requested not in INT8_RESTORE_BACKENDS:
        raise ValueError(
            f"INT8 restore backend must be one of {INT8_RESTORE_BACKENDS}"
        )
    from .triton_restore import triton_restore_available

    available = triton_restore_available(device)
    if requested == "triton":
        if not available:
            raise ValueError(
                "--int8-restore-backend triton requires CUDA and an "
                "importable Triton"
            )
        return "triton"
    if requested == "auto" and available:
        return "triton"
    return "pytorch"


def warmup_int8_restore_backend(
    requested: str,
    *,
    device,
    dtype,
    kv_heads: int,
    head_dim: int,
) -> tuple[str, float]:
    """Resolve a backend and move optional Triton JIT cost out of TTFT."""
    backend = resolve_int8_restore_backend(requested, device)
    if backend != "triton":
        return backend, 0.0
    from .triton_restore import warmup_triton_restore

    return backend, warmup_triton_restore(
        device=device,
        dtype=dtype,
        kv_heads=kv_heads,
        head_dim=head_dim,
    )


def restore_blocks(
    blocks: list[StoredBlock],
    *,
    dtype,
    device,
    int8_backend: str = "pytorch",
) -> LegacyKV:
    return restore_blocks_profiled(
        blocks,
        dtype=dtype,
        device=device,
        int8_backend=int8_backend,
    ).cache


def restore_blocks_profiled(
    blocks: list[StoredBlock],
    *,
    dtype,
    device,
    int8_backend: str = "pytorch",
) -> RestoreResult:
    """Restore blocks while separating movement, dequantization, and assembly."""
    import torch

    if not blocks:
        return RestoreResult(tuple(), "none")
    resolved_device = torch.device(device)
    if any(isinstance(block, ArenaKVBlock) for block in blocks):
        return _restore_arena_blocks(blocks, dtype=dtype, device=resolved_device)

    layered_blocks = cast(list[LayeredStoredBlock], blocks)
    quantized = _validate_object_blocks(layered_blocks)
    if quantized:
        restored_blocks, backend, transfer_s, dequant_s = _restore_quantized_blocks(
            layered_blocks,
            dtype=dtype,
            device=resolved_device,
            requested_backend=int8_backend,
        )
    else:
        restored_blocks, transfer_s = _restore_fp16_blocks(
            layered_blocks, dtype=dtype, device=resolved_device
        )
        backend = "pytorch"
        dequant_s = 0.0

    load_started = time.perf_counter()
    layers = _assemble_blocks(restored_blocks)
    _synchronize_device(resolved_device)
    return RestoreResult(
        layers,
        backend,
        load_s=time.perf_counter() - load_started,
        transfer_s=transfer_s,
        dequant_s=dequant_s,
    )


def _restore_arena_blocks(blocks, *, dtype, device) -> RestoreResult:
    if not all(isinstance(block, ArenaKVBlock) for block in blocks):
        raise TypeError("cannot mix arena-backed and object-backed KV blocks")
    load_started = time.perf_counter()
    restored = [block.restore(dtype=dtype, device=device) for block in blocks]
    cache = restored[0] if len(restored) == 1 else concatenate_caches(*restored)
    _synchronize_device(device)
    return RestoreResult(
        cache,
        "arena",
        load_s=time.perf_counter() - load_started,
    )


def _validate_object_blocks(blocks) -> bool:
    quantized = isinstance(blocks[0].layers[0][0], QuantizedTensor)
    if any(
        isinstance(key, QuantizedTensor) != quantized
        or isinstance(value, QuantizedTensor) != quantized
        for block in blocks
        for key, value in block.layers
    ):
        raise TypeError("cannot mix quantized and FP16 KV tensors")
    return quantized


def _restore_quantized_blocks(
    blocks,
    *,
    dtype,
    device,
    requested_backend: str,
):
    backend = resolve_int8_restore_backend(requested_backend, device)
    dequantize_pair = _INT8_DEQUANTIZERS[backend]

    transfer_started = time.perf_counter()
    transferred = [
        [
            (
                key.values.to(device),
                key.scales.to(device),
                value.values.to(device),
                value.scales.to(device),
            )
            for key, value in block.layers
        ]
        for block in blocks
    ]
    transfer_s = 0.0
    if device.type != "cpu":
        _synchronize_device(device)
        transfer_s = time.perf_counter() - transfer_started

    dequant_started = time.perf_counter()
    restored_blocks = [
        tuple(
            dequantize_pair(
                key_values,
                key_scales,
                value_values,
                value_scales,
                dtype=dtype,
            )
            for key_values, key_scales, value_values, value_scales in layers
        )
        for layers in transferred
    ]
    _synchronize_device(device)
    return (
        restored_blocks,
        backend,
        transfer_s,
        time.perf_counter() - dequant_started,
    )


def _restore_fp16_blocks(blocks, *, dtype, device):
    transfer_started = time.perf_counter()
    restored = [
        tuple(
            (
                key.to(device=device, dtype=dtype),
                value.to(device=device, dtype=dtype),
            )
            for key, value in block.layers
        )
        for block in blocks
    ]
    transfer_s = 0.0
    if blocks[0].layers[0][0].device != device:
        _synchronize_device(device)
        transfer_s = time.perf_counter() - transfer_started
    return restored, transfer_s


def _assemble_blocks(restored_blocks) -> LegacyKV:
    import torch

    if len(restored_blocks) == 1:
        return tuple(restored_blocks[0])
    layer_count = len(restored_blocks[0])
    return tuple(
        (
            torch.cat([block[layer][0] for block in restored_blocks], dim=-2),
            torch.cat([block[layer][1] for block in restored_blocks], dim=-2),
        )
        for layer in range(layer_count)
    )


def _pytorch_dequantize_pair(
    key_values,
    key_scales,
    value_values,
    value_scales,
    *,
    dtype,
):
    return (
        _pytorch_dequantize(key_values, key_scales, dtype=dtype),
        _pytorch_dequantize(value_values, value_scales, dtype=dtype),
    )


def _triton_dequantize_pair(
    key_values,
    key_scales,
    value_values,
    value_scales,
    *,
    dtype,
):
    from .triton_restore import dequantize_kv_int8_cuda

    return dequantize_kv_int8_cuda(
        key_values,
        key_scales,
        value_values,
        value_scales,
        dtype=dtype,
    )


def _pytorch_dequantize(values, scales, *, dtype):
    view_shape = [1] * values.ndim
    view_shape[1] = values.shape[1]
    return (values.float() * scales.view(view_shape)).to(dtype=dtype)


def _synchronize_device(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


_INT8_DEQUANTIZERS: dict[str, DequantizePair] = {
    "pytorch": _pytorch_dequantize_pair,
    "triton": _triton_dequantize_pair,
}
