"""Page-backed KV arena for atomic document allocations.

The logical cache still stores and evicts one complete document. Pages are an
allocator detail: a single generation-checked handle owns every page belonging
to that document, and releasing the handle returns all of them atomically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .page_allocator import (
    ArenaCapacityError,
    ArenaError,
    ArenaHandle,
    PageAllocator,
    StaleArenaHandle,
)


KV_BACKENDS = ("tensor", "arena")


@dataclass
class ArenaKVBlock:
    """One logical KV block backed by an arena allocation."""

    arena: "KVArena"
    handle: ArenaHandle
    token_count: int
    _released: bool = False

    @property
    def stored_bytes(self) -> int:
        return self.handle.allocated_bytes

    @property
    def useful_bytes(self) -> int:
        return self.handle.useful_bytes

    @property
    def stranded_bytes(self) -> int:
        return self.stored_bytes - self.useful_bytes

    def restore(self, *, dtype, device):
        if self._released:
            raise StaleArenaHandle("arena block has already been released")
        return self.arena.restore(self.handle, dtype=dtype, device=device)

    def release(self) -> None:
        if self._released:
            return
        self.arena.release(self.handle)
        self._released = True


class KVArena:
    """Preallocated per-layer K/V slabs with deterministic page allocation."""

    def __init__(
        self,
        template_cache,
        max_bytes: int,
        *,
        page_tokens: int = 256,
        device: Any = "cpu",
    ):
        import torch

        if max_bytes <= 0:
            raise ValueError("arena max_bytes must be positive")
        if page_tokens <= 0:
            raise ValueError("arena page_tokens must be positive")
        if not template_cache:
            raise ValueError("arena requires a non-empty KV geometry template")

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = torch.float16
        self._layer_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        bytes_per_token = 0

        expected_tokens = None
        for layer_index, (key, value) in enumerate(template_cache):
            if key.ndim < 3 or value.ndim < 3:
                raise ValueError(
                    f"arena layer {layer_index} tensors must include a token dimension"
                )
            if int(key.shape[-2]) != int(value.shape[-2]):
                raise ValueError(f"arena layer {layer_index} K/V lengths differ")
            if expected_tokens is None:
                expected_tokens = int(key.shape[-2])
            elif int(key.shape[-2]) != expected_tokens:
                raise ValueError("arena template layers have inconsistent token lengths")
            key_shape = tuple(int(part) for part in key.shape)
            value_shape = tuple(int(part) for part in value.shape)
            self._layer_shapes.append((key_shape, value_shape))
            bytes_per_token += self._tensor_bytes_per_token(key_shape)
            bytes_per_token += self._tensor_bytes_per_token(value_shape)

        self.allocator = PageAllocator(
            max_bytes,
            page_tokens=page_tokens,
            bytes_per_token=bytes_per_token,
        )
        self.arena_id = self.allocator.arena_id
        self.max_bytes = self.allocator.max_bytes
        self.page_tokens = self.allocator.page_tokens
        self.bytes_per_token = self.allocator.bytes_per_token
        self.page_bytes = self.allocator.page_bytes
        self.page_count = self.allocator.page_count
        self.reserved_bytes = self.allocator.reserved_bytes

        self.slabs = []
        for key_shape, value_shape in self._layer_shapes:
            key_page_shape = (
                self.page_count,
                *key_shape[:-2],
                self.page_tokens,
                key_shape[-1],
            )
            value_page_shape = (
                self.page_count,
                *value_shape[:-2],
                self.page_tokens,
                value_shape[-1],
            )
            self.slabs.append(
                (
                    torch.empty(key_page_shape, dtype=self.dtype, device=self.device),
                    torch.empty(value_page_shape, dtype=self.dtype, device=self.device),
                )
            )

    @staticmethod
    def _tensor_bytes_per_token(shape: tuple[int, ...]) -> int:
        # Arena storage is FP16 regardless of the model's compute dtype.
        return math.prod(shape[:-2]) * shape[-1] * 2

    def useful_bytes_for_tokens(self, token_count: int) -> int:
        return self.allocator.useful_bytes_for_tokens(token_count)

    def allocation_bytes_for_tokens(self, token_count: int) -> int:
        return self.allocator.allocation_bytes_for_tokens(token_count)

    def store(self, cache, *, owner: str) -> ArenaKVBlock:
        token_count = self._validate_cache_geometry(cache)
        handle = self.allocator.allocate(token_count, owner=owner)
        try:
            with self.torch.no_grad():
                for layer_index, (source_key, source_value) in enumerate(cache):
                    key_slab, value_slab = self.slabs[layer_index]
                    for page_offset, page_index in enumerate(handle.page_indices):
                        start = page_offset * self.page_tokens
                        end = min(start + self.page_tokens, token_count)
                        count = end - start
                        key_slab[page_index, ..., :count, :].copy_(
                            source_key[..., start:end, :].detach().to(
                                device=self.device, dtype=self.dtype
                            )
                        )
                        value_slab[page_index, ..., :count, :].copy_(
                            source_value[..., start:end, :].detach().to(
                                device=self.device, dtype=self.dtype
                            )
                        )
        except BaseException:
            self.release(handle)
            raise
        return ArenaKVBlock(self, handle, token_count)

    def restore(self, handle: ArenaHandle, *, dtype, device):
        self.allocator.validate(handle)
        layers = []
        for key_slab, value_slab in self.slabs:
            key_parts = []
            value_parts = []
            for page_offset, page_index in enumerate(handle.page_indices):
                start = page_offset * self.page_tokens
                count = min(self.page_tokens, handle.token_count - start)
                key_parts.append(key_slab[page_index, ..., :count, :])
                value_parts.append(value_slab[page_index, ..., :count, :])
            key = (
                key_parts[0].clone()
                if len(key_parts) == 1
                else self.torch.cat(key_parts, dim=-2)
            )
            value = (
                value_parts[0].clone()
                if len(value_parts) == 1
                else self.torch.cat(value_parts, dim=-2)
            )
            layers.append(
                (
                    key.to(device=device, dtype=dtype),
                    value.to(device=device, dtype=dtype),
                )
            )
        return tuple(layers)

    def release(self, handle: ArenaHandle) -> None:
        self.allocator.release(handle)

    def _validate_cache_geometry(self, cache) -> int:
        if len(cache) != len(self._layer_shapes):
            raise ValueError("arena cache layer count does not match its template")
        token_count = None
        for layer_index, ((key, value), (key_shape, value_shape)) in enumerate(
            zip(cache, self._layer_shapes)
        ):
            if tuple(key.shape[:-2]) + (int(key.shape[-1]),) != (
                key_shape[:-2] + (key_shape[-1],)
            ):
                raise ValueError(f"arena key geometry mismatch at layer {layer_index}")
            if tuple(value.shape[:-2]) + (int(value.shape[-1]),) != (
                value_shape[:-2] + (value_shape[-1],)
            ):
                raise ValueError(f"arena value geometry mismatch at layer {layer_index}")
            if int(key.shape[-2]) != int(value.shape[-2]):
                raise ValueError(f"arena K/V token mismatch at layer {layer_index}")
            if token_count is None:
                token_count = int(key.shape[-2])
            elif int(key.shape[-2]) != token_count:
                raise ValueError("arena cache layers have inconsistent token lengths")
        if token_count is None or token_count <= 0:
            raise ValueError("arena cannot store an empty cache")
        return token_count

    def stats(self) -> dict[str, int | str]:
        return self.allocator.stats()

    def assert_consistent(self) -> None:
        self.allocator.assert_consistent()

    @property
    def live_allocated_bytes(self) -> int:
        return self.allocator.live_allocated_bytes

    @property
    def live_useful_bytes(self) -> int:
        return self.allocator.live_useful_bytes

    @property
    def peak_allocated_bytes(self) -> int:
        return self.allocator.peak_allocated_bytes
