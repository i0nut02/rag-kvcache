"""KV tensor storage, quantization, slicing, and restoration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .arena import ArenaKVBlock


LegacyKV = tuple[tuple[Any, Any], ...]
STORAGE_MODES = ("accelerator-fp16", "cpu-fp16", "cpu-int8")
INT8_RESTORE_BACKENDS = ("pytorch", "auto", "triton")


@dataclass
class RestoreResult:
    cache: LegacyKV
    backend: str
    load_s: float = 0.0
    transfer_s: float = 0.0
    dequant_s: float = 0.0


@dataclass
class QuantizedTensor:
    values: Any
    scales: Any

    @property
    def stored_bytes(self) -> int:
        return tensor_bytes(self.values) + tensor_bytes(self.scales)


@dataclass
class KVBlock:
    layers: tuple[tuple[Any, Any], ...]
    token_count: int

    @property
    def stored_bytes(self) -> int:
        total = 0
        for key, value in self.layers:
            total += object_bytes(key) + object_bytes(value)
        return total

    @property
    def useful_bytes(self) -> int:
        return self.stored_bytes

    @property
    def stranded_bytes(self) -> int:
        return 0

    def release(self) -> None:
        """Object-backed tensors are released when their owner drops them."""


def tensor_bytes(tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def object_bytes(value) -> int:
    return value.stored_bytes if isinstance(value, QuantizedTensor) else tensor_bytes(value)


def to_legacy(cache) -> LegacyKV:
    if isinstance(cache, tuple):
        return cache
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    # Transformers 5 stores each K/V pair in a CacheLayer. Its DynamicCache
    # iterator yields a third sliding-window value, so read the layer tensors
    # explicitly instead of converting the iterator directly to a tuple.
    if hasattr(cache, "layers"):
        result = []
        for index, layer in enumerate(cache.layers):
            key = getattr(layer, "keys", None)
            value = getattr(layer, "values", None)
            if key is None or value is None:
                raise TypeError(
                    f"cache layer {index} is not initialized or has no K/V tensors"
                )
            result.append((key, value))
        return tuple(result)
    if hasattr(cache, "key_cache"):
        return tuple(zip(cache.key_cache, cache.value_cache))
    raise TypeError(f"unsupported cache type: {type(cache)!r}")


def to_model_cache(cache: LegacyKV):
    """Give Transformers an owned dynamic cache so serving cannot mutate L0."""
    try:
        from transformers import DynamicCache

        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(cache)
        # Transformers 5 removed from_legacy_cache. The first constructor
        # argument accepts the same per-layer (key, value) iterable.
        return DynamicCache(cache)
    except (ImportError, AttributeError):
        return cache


def sequence_length(cache: LegacyKV) -> int:
    return 0 if not cache else int(cache[0][0].shape[-2])


def slice_cache(cache: LegacyKV, start: int, end: int) -> LegacyKV:
    return tuple(
        (key[..., start:end, :].contiguous(), value[..., start:end, :].contiguous())
        for key, value in cache
    )


def split_blocks(cache: LegacyKV, block_tokens: int) -> list[KVBlock]:
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive")
    length = sequence_length(cache)
    return [
        KVBlock(slice_cache(cache, start, min(start + block_tokens, length)), min(block_tokens, length - start))
        for start in range(0, length, block_tokens)
    ]


def symmetric_int8(tensor) -> QuantizedTensor:
    """Quantize independently for every layer/KV-head tensor passed here."""
    import torch

    if tensor.ndim < 3:
        raise ValueError("KV tensors must include batch, head, and feature dimensions")
    reduce_dims = tuple(index for index in range(tensor.ndim) if index != 1)
    maxima = tensor.float().abs().amax(dim=reduce_dims)
    scales = (maxima / 127.0).clamp_min(torch.finfo(torch.float32).eps)
    view_shape = [1] * tensor.ndim
    view_shape[1] = tensor.shape[1]
    values = torch.round(tensor.float() / scales.view(view_shape)).clamp(-127, 127).to(torch.int8).cpu()
    return QuantizedTensor(values=values, scales=scales.to(torch.float32).cpu())


def dequantize(value: QuantizedTensor, *, dtype, device):
    view_shape = [1] * value.values.ndim
    view_shape[1] = value.values.shape[1]
    restored = value.values.float() * value.scales.view(view_shape)
    return restored.to(device=device, dtype=dtype)


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
                "--int8-restore-backend triton requires CUDA and an importable Triton"
            )
        return "triton"
    if requested == "auto" and available:
        return "triton"
    return "pytorch"


def resolve_storage_device(mode: str, accelerator_device=None):
    if mode not in STORAGE_MODES:
        raise ValueError(f"storage mode must be one of {STORAGE_MODES}")
    if mode != "accelerator-fp16":
        return "cpu"
    if accelerator_device is None:
        raise ValueError("accelerator-fp16 storage requires a CUDA or MPS device")
    device_type = getattr(accelerator_device, "type", str(accelerator_device).split(":", 1)[0])
    if device_type not in {"cuda", "mps"}:
        raise ValueError("accelerator-fp16 storage requires a CUDA or MPS device")
    return accelerator_device


def store_blocks(
    cache: LegacyKV,
    mode: str,
    block_tokens: int,
    *,
    accelerator_device=None,
) -> list[KVBlock]:
    import torch

    target_device = resolve_storage_device(mode, accelerator_device)
    raw_blocks = split_blocks(cache, block_tokens)
    stored = []
    for block in raw_blocks:
        layers = []
        for key, value in block.layers:
            if mode == "cpu-int8":
                layers.append((symmetric_int8(key), symmetric_int8(value)))
            else:
                layers.append(
                    (
                        key.detach().to(device=target_device, dtype=torch.float16).contiguous(),
                        value.detach().to(device=target_device, dtype=torch.float16).contiguous(),
                    )
                )
        stored.append(KVBlock(tuple(layers), block.token_count))
    return stored


def restore_blocks(
    blocks: list[Any],
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
    blocks: list[Any],
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
        if not all(isinstance(block, ArenaKVBlock) for block in blocks):
            raise TypeError("cannot mix arena-backed and object-backed KV blocks")
        load_started = time.perf_counter()
        restored = [block.restore(dtype=dtype, device=device) for block in blocks]
        cache = restored[0] if len(restored) == 1 else concatenate_caches(*restored)
        _synchronize_device(resolved_device)
        return RestoreResult(
            cache,
            "arena",
            load_s=time.perf_counter() - load_started,
        )

    layer_count = len(blocks[0].layers)
    quantized = isinstance(blocks[0].layers[0][0], QuantizedTensor)
    if any(
        isinstance(key, QuantizedTensor) != quantized
        or isinstance(value, QuantizedTensor) != quantized
        for block in blocks
        for key, value in block.layers
    ):
        raise TypeError("cannot mix quantized and FP16 KV tensors")

    transfer_s = 0.0
    dequant_s = 0.0
    backend = "pytorch"
    if quantized:
        backend = resolve_int8_restore_backend(int8_backend, resolved_device)
        transferred = []
        transfer_started = time.perf_counter()
        for block in blocks:
            transferred_layers = []
            for key, value in block.layers:
                transferred_layers.append(
                    (
                        key.values.to(resolved_device),
                        key.scales.to(resolved_device),
                        value.values.to(resolved_device),
                        value.scales.to(resolved_device),
                    )
                )
            transferred.append(transferred_layers)
        if resolved_device.type != "cpu":
            _synchronize_device(resolved_device)
            transfer_s = time.perf_counter() - transfer_started

        dequant_started = time.perf_counter()
        restored_blocks = []
        for transferred_layers in transferred:
            restored_layers = []
            for key_values, key_scales, value_values, value_scales in transferred_layers:
                if backend == "triton":
                    from .triton_restore import dequantize_kv_int8_cuda

                    key, value = dequantize_kv_int8_cuda(
                        key_values,
                        key_scales,
                        value_values,
                        value_scales,
                        dtype=dtype,
                    )
                else:
                    key = _dequantize_device(
                        key_values, key_scales, dtype=dtype
                    )
                    value = _dequantize_device(
                        value_values, value_scales, dtype=dtype
                    )
                restored_layers.append((key, value))
            restored_blocks.append(tuple(restored_layers))
        _synchronize_device(resolved_device)
        dequant_s = time.perf_counter() - dequant_started
    else:
        transfer_started = time.perf_counter()
        restored_blocks = [
            tuple(
                (
                    key.to(device=resolved_device, dtype=dtype),
                    value.to(device=resolved_device, dtype=dtype),
                )
                for key, value in block.layers
            )
            for block in blocks
        ]
        source_device = blocks[0].layers[0][0].device
        if source_device != resolved_device:
            _synchronize_device(resolved_device)
            transfer_s = time.perf_counter() - transfer_started

    load_started = time.perf_counter()
    if len(restored_blocks) == 1:
        layers = restored_blocks[0]
    else:
        layers = []
        for layer_index in range(layer_count):
            keys = [block[layer_index][0] for block in restored_blocks]
            values = [block[layer_index][1] for block in restored_blocks]
            layers.append((torch.cat(keys, dim=-2), torch.cat(values, dim=-2)))
    _synchronize_device(resolved_device)
    return RestoreResult(
        tuple(layers),
        backend,
        load_s=time.perf_counter() - load_started,
        transfer_s=transfer_s,
        dequant_s=dequant_s,
    )


def _dequantize_device(values, scales, *, dtype):
    view_shape = [1] * values.ndim
    view_shape[1] = values.shape[1]
    return (values.float() * scales.view(view_shape)).to(dtype=dtype)


def _synchronize_device(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def slice_stored_blocks(blocks: list[Any], start: int, end: int) -> list[KVBlock]:
    """Slice stored KV blocks without restoring or changing their storage mode."""
    if start < 0 or end < start:
        raise ValueError("invalid stored-block slice")
    if any(isinstance(block, ArenaKVBlock) for block in blocks):
        raise ValueError(
            "arena allocations are atomic document objects and cannot be sliced"
        )
    result = []
    offset = 0
    for block in blocks:
        block_end = offset + block.token_count
        overlap_start = max(start, offset)
        overlap_end = min(end, block_end)
        if overlap_start < overlap_end:
            local_start = overlap_start - offset
            local_end = overlap_end - offset
            layers = []
            for key, value in block.layers:
                layers.append(
                    (
                        _slice_stored_tensor(key, local_start, local_end),
                        _slice_stored_tensor(value, local_start, local_end),
                    )
                )
            result.append(KVBlock(tuple(layers), overlap_end - overlap_start))
        offset = block_end
        if offset >= end:
            break
    if sum(block.token_count for block in result) != end - start:
        raise ValueError("stored blocks do not cover the requested slice")
    return result


def _slice_stored_tensor(value, start: int, end: int):
    if isinstance(value, QuantizedTensor):
        return QuantizedTensor(
            values=value.values[..., start:end, :].contiguous(),
            scales=value.scales.clone(),
        )
    return value[..., start:end, :].contiguous()


def release_blocks(blocks: list[Any]) -> None:
    """Release external allocations, then make every block unreachable."""
    for block in list(blocks):
        release = getattr(block, "release", None)
        if release is not None:
            release()
    blocks.clear()


def concatenate_caches(*caches: LegacyKV) -> LegacyKV:
    import torch

    present = [cache for cache in caches if cache]
    if not present:
        return tuple()
    return tuple(
        (
            torch.cat([cache[layer][0] for cache in present], dim=-2),
            torch.cat([cache[layer][1] for cache in present], dim=-2),
        )
        for layer in range(len(present[0]))
    )
