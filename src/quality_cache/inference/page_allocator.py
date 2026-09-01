"""Deterministic, generation-checked page allocation without tensor ownership."""

from __future__ import annotations

import heapq
import itertools
import sys
from dataclasses import dataclass


class ArenaError(RuntimeError):
    """Base error for arena allocation and handle failures."""


class ArenaCapacityError(ArenaError):
    """Raised when an allocation cannot fit in the configured arena."""


class StaleArenaHandle(ArenaError):
    """Raised when a freed, replaced, malformed, or foreign handle is used."""


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


_ALLOCATOR_IDS = itertools.count(1)


class PageAllocator:
    """Own page lifetimes while leaving physical storage to the KV arena.

    Lowest-numbered free pages are selected first, making allocation and tests
    deterministic. A handle captures each page generation so stale or foreign
    references cannot access storage after release and reuse.
    """

    def __init__(
        self,
        max_bytes: int,
        *,
        page_tokens: int,
        bytes_per_token: int,
    ):
        if max_bytes <= 0:
            raise ValueError("arena max_bytes must be positive")
        if page_tokens <= 0:
            raise ValueError("arena page_tokens must be positive")
        if bytes_per_token <= 0:
            raise ValueError("arena bytes_per_token must be positive")

        self.arena_id = next(_ALLOCATOR_IDS)
        self.max_bytes = int(max_bytes)
        self.page_tokens = int(page_tokens)
        self.bytes_per_token = int(bytes_per_token)
        self.page_bytes = self.bytes_per_token * self.page_tokens
        self.page_count = self.max_bytes // self.page_bytes
        if self.page_count <= 0:
            raise ArenaCapacityError(
                f"arena budget {self.max_bytes} cannot hold one "
                f"{self.page_bytes}-byte page"
            )
        self.reserved_bytes = self.page_count * self.page_bytes

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

    def useful_bytes_for_tokens(self, token_count: int) -> int:
        if token_count < 0:
            raise ValueError("token_count must be nonnegative")
        return token_count * self.bytes_per_token

    def allocation_bytes_for_tokens(self, token_count: int) -> int:
        if token_count <= 0:
            raise ValueError("arena allocations require at least one token")
        pages = (token_count + self.page_tokens - 1) // self.page_tokens
        return pages * self.page_bytes

    def allocate(self, token_count: int, *, owner: str) -> ArenaHandle:
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
        self._assert_fast_consistent()
        return handle

    def release(self, handle: ArenaHandle) -> None:
        self.validate(handle)
        self._allocations.pop(handle.allocation_id)
        for page_index in handle.page_indices:
            self._page_generations[page_index] += 1
            heapq.heappush(self._free_pages, page_index)
        self.live_allocated_bytes -= handle.allocated_bytes
        self.live_useful_bytes -= handle.useful_bytes
        self.releases += 1
        self._assert_fast_consistent()

    def validate(self, handle: ArenaHandle) -> None:
        valid = isinstance(handle, ArenaHandle)
        if valid:
            valid = (
                handle.arena_id == self.arena_id
                and len(handle.page_indices) == len(handle.page_generations)
                and all(
                    isinstance(index, int) and 0 <= index < self.page_count
                    for index in handle.page_indices
                )
                and self._allocations.get(handle.allocation_id) == handle
            )
        if valid:
            valid = all(
                self._page_generations[index] == generation
                for index, generation in zip(
                    handle.page_indices, handle.page_generations
                )
            )
        if not valid:
            self.stale_rejections += 1
            allocation_id = getattr(handle, "allocation_id", "unknown")
            raise StaleArenaHandle(
                f"arena allocation {allocation_id} is stale, malformed, or foreign"
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

    def assert_consistent(self) -> None:
        """Run the exhaustive page-ownership audit used by tests/debugging."""
        self._assert_fast_consistent()
        allocated_pages = [
            index
            for handle in self._allocations.values()
            for index in handle.page_indices
        ]
        free_pages = set(self._free_pages)
        assert len(free_pages) == len(self._free_pages)
        assert len(set(allocated_pages)) == len(allocated_pages)
        assert free_pages.isdisjoint(allocated_pages)
        assert free_pages | set(allocated_pages) == set(range(self.page_count))
        assert len(self._page_generations) == self.page_count
        assert all(
            self._free_pages[(index - 1) // 2] <= self._free_pages[index]
            for index in range(1, len(self._free_pages))
        )
        assert all(
            allocation_id == handle.allocation_id
            and handle.arena_id == self.arena_id
            and handle.allocated_tokens
            == len(handle.page_indices) * self.page_tokens
            and handle.allocated_bytes == len(handle.page_indices) * self.page_bytes
            and handle.useful_bytes
            == self.useful_bytes_for_tokens(handle.token_count)
            for allocation_id, handle in self._allocations.items()
        )
        assert self.live_allocated_bytes == sum(
            handle.allocated_bytes for handle in self._allocations.values()
        )
        assert self.live_useful_bytes == sum(
            handle.useful_bytes for handle in self._allocations.values()
        )

    def _assert_fast_consistent(self) -> None:
        assert 0 <= self.live_useful_bytes <= self.live_allocated_bytes
        assert self.live_allocated_bytes <= self.reserved_bytes <= self.max_bytes
        assert self.live_allocated_bytes == (
            self.page_count - len(self._free_pages)
        ) * self.page_bytes
