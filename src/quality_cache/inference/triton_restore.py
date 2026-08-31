"""Optional fused CUDA INT8 KV dequantization kernel.

The module imports Triton opportunistically. CPU, MPS, and clean environments
without Triton retain a fully supported PyTorch implementation in tensors.py.
"""

from __future__ import annotations

from typing import Any


class TritonRestoreUnavailable(RuntimeError):
    """Raised when an explicitly requested Triton restore cannot run."""


try:  # Triton is distributed with CUDA PyTorch builds, but not CPU/MPS builds.
    import triton
    import triton.language as tl
except (ImportError, RuntimeError, OSError):  # pragma: no cover - platform dependent
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _dequantize_kv_kernel(
        key_values,
        key_scales,
        value_values,
        value_scales,
        key_output,
        value_output,
        element_count,
        HEAD_STRIDE: tl.constexpr,
        HEAD_COUNT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < element_count
        heads = (offsets // HEAD_STRIDE) % HEAD_COUNT
        key = tl.load(key_values + offsets, mask=mask, other=0.0).to(tl.float32)
        value = tl.load(value_values + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        key_scale = tl.load(key_scales + heads, mask=mask, other=0.0)
        value_scale = tl.load(value_scales + heads, mask=mask, other=0.0)
        # Destination pointer types perform the final FP16/BF16 cast. There is
        # no full-sized FP32 intermediate tensor.
        tl.store(key_output + offsets, key * key_scale, mask=mask)
        tl.store(value_output + offsets, value * value_scale, mask=mask)

else:  # pragma: no cover - exercised indirectly by fallback tests
    _dequantize_kv_kernel = None


def triton_restore_available(device: Any = "cuda") -> bool:
    """Return whether this process can launch the CUDA restore kernel."""
    if triton is None or _dequantize_kv_kernel is None:
        return False
    try:
        import torch

        resolved = torch.device(device)
        return resolved.type == "cuda" and torch.cuda.is_available()
    except (ImportError, RuntimeError, TypeError):
        return False


def dequantize_kv_int8_cuda(
    key_values,
    key_scales,
    value_values,
    value_scales,
    *,
    dtype,
):
    """Dequantize one K/V tensor pair already resident on CUDA."""
    import torch

    if not triton_restore_available(key_values.device):
        raise TritonRestoreUnavailable(
            "Triton INT8 restore requires a CUDA device and an importable Triton"
        )
    tensors = (key_values, key_scales, value_values, value_scales)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise TritonRestoreUnavailable("Triton INT8 restore inputs must be on CUDA")
    if key_values.dtype != torch.int8 or value_values.dtype != torch.int8:
        raise TypeError("Triton INT8 restore values must use torch.int8")
    if key_scales.dtype != torch.float32 or value_scales.dtype != torch.float32:
        raise TypeError("Triton INT8 restore scales must use torch.float32")
    if key_values.shape != value_values.shape:
        raise ValueError("Triton INT8 restore requires matching K/V shapes")
    if key_values.ndim != 4:
        raise ValueError("Triton restore expects KV tensors shaped [B, H, T, D]")
    if key_scales.numel() != key_values.shape[1]:
        raise ValueError("key scale count must equal the number of KV heads")
    if value_scales.numel() != value_values.shape[1]:
        raise ValueError("value scale count must equal the number of KV heads")
    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError("Triton restore output must be float16, bfloat16, or float32")

    key_values = key_values.contiguous()
    value_values = value_values.contiguous()
    key_scales = key_scales.contiguous()
    value_scales = value_scales.contiguous()
    key_output = torch.empty_like(key_values, dtype=dtype)
    value_output = torch.empty_like(value_values, dtype=dtype)
    element_count = key_values.numel()
    head_stride = element_count // (
        int(key_values.shape[0]) * int(key_values.shape[1])
    )
    block_size = 256
    _dequantize_kv_kernel[(triton.cdiv(element_count, block_size),)](
        key_values,
        key_scales,
        value_values,
        value_scales,
        key_output,
        value_output,
        element_count,
        HEAD_STRIDE=head_stride,
        HEAD_COUNT=int(key_values.shape[1]),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return key_output, value_output
