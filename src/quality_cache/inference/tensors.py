"""KV tensor representation, storage, quantization, and slicing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from .arena import ArenaKVBlock
from .kv_types import LayeredStoredBlock, LegacyKV, StoredBlock


STORAGE_MODES = ("accelerator-fp16", "cpu-fp16", "cpu-int8")


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


def slice_stored_blocks(
    blocks: list[StoredBlock], start: int, end: int
) -> list[KVBlock]:
    """Slice stored KV blocks without restoring or changing their storage mode."""
    if start < 0 or end < start:
        raise ValueError("invalid stored-block slice")
    if any(isinstance(block, ArenaKVBlock) for block in blocks):
        raise ValueError(
            "arena allocations are atomic document objects and cannot be sliced"
        )
    layered_blocks = cast(list[LayeredStoredBlock], blocks)
    result = []
    offset = 0
    for block in layered_blocks:
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


def release_blocks(blocks: list[StoredBlock]) -> None:
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
