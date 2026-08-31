"""Page-backed KV arena for atomic document allocations.

The logical cache still stores and evicts one complete document. Pages are an
allocator detail: a single generation-checked handle owns every page belonging
to that document, and releasing the handle returns all of them atomically.
"""

from __future__ import annotations

import heapq
import itertools
import math
import sys
from dataclasses import dataclass
from typing import Any


KV_BACKENDS = ("tensor", "arena")


class ArenaError(RuntimeError):
    """Base error for arena allocation and handle failures."""


class ArenaCapacityError(ArenaError):
    """Raised when an allocation cannot fit in the configured arena."""


class StaleArenaHandle(ArenaError):
    """Raised when a freed, replaced, or foreign allocation is accessed."""


@dataclass(frozen=True)
class ArenaHandle:
    arena_id: int
    allocation_id: int
    page_indices: tuple[int, ...]
    page_generations: tuple[int, ...]
    token_count: int
    allocated_tokens: int
    useful_bytes: int
    allocated_bytes: int
    owner: str


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


_ARENA_IDS = itertools.count(1)


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
        self.arena_id = next(_ARENA_IDS)
        self.max_bytes = int(max_bytes)
        self.page_tokens = int(page_tokens)
        self.device = torch.device(device)
        self.dtype = torch.float16
        self._layer_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.bytes_per_token = 0

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
            self.bytes_per_token += self._tensor_bytes_per_token(key_shape)
            self.bytes_per_token += self._tensor_bytes_per_token(value_shape)

        self.page_bytes = self.bytes_per_token * self.page_tokens
        self.page_count = self.max_bytes // self.page_bytes
        if self.page_count <= 0:
            raise ArenaCapacityError(
                f"arena budget {self.max_bytes} cannot hold one {self.page_bytes}-byte page"
            )
        self.reserved_bytes = self.page_count * self.page_bytes

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

        self._free_pages = list(range(self.page_count))
        heapq.heapify(self._free_pages)
        self._page_generations = [0] * self.page_count
        self._allocations: dict[int, ArenaHandle] = {}
        self._next_allocation_id = 1
        self.live_allocated_bytes = 0
        self.live_useful_bytes = 0
        self.peak_allocated_bytes = 0
        self.allocations = 0
        self.releases = 0
        self.stale_rejections = 0

    @staticmethod
    def _tensor_bytes_per_token(shape: tuple[int, ...]) -> int:
        # Arena storage is FP16 regardless of the model's compute dtype.
        return math.prod(shape[:-2]) * shape[-1] * 2

    def useful_bytes_for_tokens(self, token_count: int) -> int:
        if token_count < 0:
            raise ValueError("token_count must be nonnegative")
        return token_count * self.bytes_per_token

    def allocation_bytes_for_tokens(self, token_count: int) -> int:
        if token_count <= 0:
            raise ValueError("arena allocations require at least one token")
        pages = (token_count + self.page_tokens - 1) // self.page_tokens
        return pages * self.page_bytes

    def store(self, cache, *, owner: str) -> ArenaKVBlock:
        token_count = self._validate_cache_geometry(cache)
        handle = self._allocate(token_count, owner=owner)
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
        self._validate_handle(handle)
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
        self._validate_handle(handle)
        self._allocations.pop(handle.allocation_id)
        for page_index in handle.page_indices:
            self._page_generations[page_index] += 1
            heapq.heappush(self._free_pages, page_index)
        self.live_allocated_bytes -= handle.allocated_bytes
        self.live_useful_bytes -= handle.useful_bytes
        self.releases += 1
        self._assert_consistent()

    def _allocate(self, token_count: int, *, owner: str) -> ArenaHandle:
        allocated_bytes = self.allocation_bytes_for_tokens(token_count)
        pages_needed = allocated_bytes // self.page_bytes
        if pages_needed > len(self._free_pages):
            raise ArenaCapacityError(
                f"arena needs {pages_needed} pages for {token_count} tokens but "
                f"only {len(self._free_pages)} are free"
            )
        pages = tuple(heapq.heappop(self._free_pages) for _ in range(pages_needed))
        generations = tuple(self._page_generations[index] for index in pages)
        allocation_id = self._next_allocation_id
        self._next_allocation_id += 1
        handle = ArenaHandle(
            arena_id=self.arena_id,
            allocation_id=allocation_id,
            page_indices=pages,
            page_generations=generations,
            token_count=token_count,
            allocated_tokens=pages_needed * self.page_tokens,
            useful_bytes=self.useful_bytes_for_tokens(token_count),
            allocated_bytes=allocated_bytes,
            owner=str(owner),
        )
        self._allocations[allocation_id] = handle
        self.live_allocated_bytes += allocated_bytes
        self.live_useful_bytes += handle.useful_bytes
        self.peak_allocated_bytes = max(
            self.peak_allocated_bytes, self.live_allocated_bytes
        )
        self.allocations += 1
        self._assert_consistent()
        return handle

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

    def _validate_handle(self, handle: ArenaHandle) -> None:
        valid = True
        if handle.arena_id != self.arena_id:
            valid = False
        elif self._allocations.get(handle.allocation_id) != handle:
            valid = False
        elif any(
            self._page_generations[index] != generation
            for index, generation in zip(
                handle.page_indices, handle.page_generations
            )
        ):
            valid = False
        if not valid:
            self.stale_rejections += 1
            raise StaleArenaHandle(
                f"arena allocation {handle.allocation_id} is stale or foreign"
            )

    def stats(self) -> dict[str, int | str]:
        metadata_bytes = (
            sys.getsizeof(self._free_pages)
            + sys.getsizeof(self._page_generations)
            + sys.getsizeof(self._allocations)
            + sum(sys.getsizeof(handle) for handle in self._allocations.values())
        )
        return {
            "kv_backend": "arena",
            "arena_page_tokens": self.page_tokens,
            "arena_pages_total": self.page_count,
            "arena_pages_free": len(self._free_pages),
            "arena_live_allocations": len(self._allocations),
            "arena_reserved_bytes": self.reserved_bytes,
            "arena_free_bytes": self.reserved_bytes - self.live_allocated_bytes,
            "arena_peak_allocated_bytes": self.peak_allocated_bytes,
            "arena_metadata_bytes": metadata_bytes,
            "arena_allocations": self.allocations,
            "arena_releases": self.releases,
            "arena_stale_rejections": self.stale_rejections,
            "arena_useful_bytes": self.live_useful_bytes,
            "arena_stranded_bytes": (
                self.live_allocated_bytes - self.live_useful_bytes
            ),
        }

    def _assert_consistent(self) -> None:
        assert 0 <= self.live_useful_bytes <= self.live_allocated_bytes
        assert self.live_allocated_bytes <= self.reserved_bytes <= self.max_bytes
        assert len(self._free_pages) + sum(
            len(handle.page_indices) for handle in self._allocations.values()
        ) == self.page_count
        assert len(set(self._free_pages)) == len(self._free_pages)
