"""Content-addressed fixed-size prefix-block cache strategy."""

from __future__ import annotations

import hashlib
import heapq
import sys
from dataclasses import dataclass
from typing import Any

from .article import POLICIES, CacheKey
from .base import PrefixLookup, StoredKV, cache_namespace


@dataclass
class _FixedEntry:
    digest: str
    parent: str
    tokens: tuple[int, ...]
    token_count: int
    payload: StoredKV
    prefill_cost_s: float
    inserted_at: int
    last_access: int
    frequency: int = 1
    priority: float = 0.0
    heap_version: int = 0


class FixedBlockPrefixCache:
    """Content-addressed, independently evicted, vLLM-style prefix blocks."""

    strategy = "fixed-block"

    def __init__(
        self,
        max_bytes: int,
        *,
        block_tokens: int = 16,
        policy: str = "lru",
        l0=None,
    ):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive")
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self.max_bytes = int(max_bytes)
        self.block_tokens = block_tokens
        self.policy = policy
        self.l0 = l0
        self.entries: dict[str, _FixedEntry] = {}
        self.root_hashes: set[str] = set()
        self._child_counts: dict[str, int] = {}
        self._victim_heap: list[tuple[tuple[Any, ...], int, str]] = []
        self._entry_metadata_bytes = 0
        self._cached_tokens = 0
        self.current_bytes = 0
        self.clock = 0
        self.gdsf_clock = 0.0
        self.insertions = 0
        self.evictions = 0

    def __len__(self):
        return len(self.entries)

    def _root(self, key: CacheKey) -> str:
        root = cache_namespace(key)
        if root not in self.root_hashes:
            self.root_hashes.add(root)
        return root

    @staticmethod
    def _digest(parent: str, tokens: tuple[int, ...]) -> str:
        raw = parent.encode("ascii") + b"\0" + b",".join(
            str(token).encode("ascii") for token in tokens
        )
        return hashlib.sha256(raw).hexdigest()

    def lookup(self, key: CacheKey, tokens: list[int]) -> PrefixLookup:
        self.clock += 1
        parent = self._root(key)
        payloads = []
        matched = 0
        for start in range(0, len(tokens) - self.block_tokens + 1, self.block_tokens):
            block_tokens = tuple(tokens[start : start + self.block_tokens])
            digest = self._digest(parent, block_tokens)
            entry = self.entries.get(digest)
            if entry is None or entry.parent != parent or entry.tokens != block_tokens:
                break
            entry.frequency += 1
            entry.last_access = self.clock
            if self.policy == "gdsf":
                entry.priority = self._gdsf_priority(entry)
            self._refresh_leaf(entry)
            payloads.append(entry.payload)
            matched += entry.token_count
            parent = digest
        return PrefixLookup(matched, payloads, len(tokens))

    def insert(
        self, key: CacheKey, tokens: list[int], payload: StoredKV, prefill_cost_s: float
    ) -> bool:
        parent = self._root(key)
        admitted = False
        protected: set[str] = set()
        for start in range(0, len(tokens) - self.block_tokens + 1, self.block_tokens):
            block_tokens = tuple(tokens[start : start + self.block_tokens])
            digest = self._digest(parent, block_tokens)
            self.clock += 1
            existing = self.entries.get(digest)
            if existing is not None:
                existing.last_access = self.clock
                existing.heap_version += 1
                protected.add(digest)
                parent = digest
                admitted = True
                continue
            block_payload = payload.slice(start, start + self.block_tokens)
            size = block_payload.stored_bytes
            if size > self.max_bytes:
                parent = digest
                continue
            while self.current_bytes + size > self.max_bytes:
                if not self._evict_one(protected):
                    self._schedule_protected_leaves(protected)
                    self._assert_budget()
                    return admitted
            entry = _FixedEntry(
                digest=digest,
                parent=parent,
                tokens=block_tokens,
                token_count=len(block_tokens),
                payload=block_payload,
                prefill_cost_s=prefill_cost_s
                * len(block_tokens)
                / max(1, len(tokens)),
                inserted_at=self.clock,
                last_access=self.clock,
            )
            entry.priority = self._gdsf_priority(entry)
            parent_entry = self.entries.get(parent)
            if parent_entry is not None:
                self._child_counts[parent] += 1
                # A node with a child is no longer an eviction candidate.
                parent_entry.heap_version += 1
            self.entries[digest] = entry
            self._child_counts[digest] = 0
            self.current_bytes += size
            self._cached_tokens += entry.token_count
            self._entry_metadata_bytes += self._entry_metadata_size(entry)
            self.insertions += 1
            admitted = True
            protected.add(digest)
            parent = digest
        self._schedule_protected_leaves(protected)
        self._maybe_compact_heap()
        self._assert_budget()
        return admitted

    def _evict_one(self, protected: set[str] | None = None) -> bool:
        protected = protected or set()
        victim = self._pop_victim(protected)
        if victim is None:
            return False
        self.entries.pop(victim.digest)
        self._child_counts.pop(victim.digest)
        self.current_bytes -= victim.payload.stored_bytes
        self._cached_tokens -= victim.token_count
        self._entry_metadata_bytes -= self._entry_metadata_size(victim)
        if self.policy == "gdsf":
            self.gdsf_clock = victim.priority
        parent = self.entries.get(victim.parent)
        if parent is not None:
            remaining = self._child_counts[victim.parent] - 1
            self._child_counts[victim.parent] = remaining
            if remaining == 0:
                self._schedule_leaf(parent)
        victim.payload.blocks.clear()
        self.evictions += 1
        return True

    def _victim_priority(self, entry: _FixedEntry) -> tuple[Any, ...]:
        if self.policy == "lru":
            return (entry.last_access, entry.inserted_at)
        if self.policy == "lfu":
            return (entry.frequency, entry.last_access, entry.inserted_at)
        if self.policy == "fifo":
            return (entry.inserted_at,)
        return (entry.priority, entry.last_access, entry.inserted_at)

    def _schedule_leaf(self, entry: _FixedEntry) -> None:
        if self._child_counts.get(entry.digest) != 0:
            return
        entry.heap_version += 1
        heapq.heappush(
            self._victim_heap,
            (self._victim_priority(entry), entry.heap_version, entry.digest),
        )

    def _refresh_leaf(self, entry: _FixedEntry) -> None:
        if self._child_counts.get(entry.digest) == 0:
            self._schedule_leaf(entry)

    def _schedule_protected_leaves(self, protected: set[str]) -> None:
        for digest in protected:
            entry = self.entries.get(digest)
            if entry is not None and self._child_counts.get(digest) == 0:
                self._schedule_leaf(entry)

    def _pop_victim(self, protected: set[str]) -> _FixedEntry | None:
        deferred: list[tuple[tuple[Any, ...], int, str]] = []
        victim = None
        while self._victim_heap:
            item = heapq.heappop(self._victim_heap)
            _, version, digest = item
            entry = self.entries.get(digest)
            if (
                entry is None
                or entry.heap_version != version
                or self._child_counts.get(digest) != 0
            ):
                continue
            if digest in protected:
                deferred.append(item)
                continue
            victim = entry
            break
        for item in deferred:
            heapq.heappush(self._victim_heap, item)
        return victim

    def _maybe_compact_heap(self) -> None:
        # Access updates are appended lazily. Rebuild only when stale records
        # substantially outnumber the cache entries, keeping normal eviction
        # O(log B) while bounding policy metadata.
        if len(self._victim_heap) <= max(1024, 4 * len(self.entries)):
            return
        self._victim_heap.clear()
        for entry in self.entries.values():
            if self._child_counts.get(entry.digest) == 0:
                self._schedule_leaf(entry)

    def _gdsf_priority(self, entry: _FixedEntry) -> float:
        return (
            self.gdsf_clock
            + entry.frequency
            * max(entry.prefill_cost_s, 1e-12)
            / max(entry.payload.stored_bytes, 1)
        )

    def stats(self) -> dict[str, Any]:
        metadata_bytes = (
            sys.getsizeof(self.entries)
            + sys.getsizeof(self.root_hashes)
            + sys.getsizeof(self._child_counts)
            + len(self._child_counts) * sys.getsizeof(0)
            + sys.getsizeof(self._victim_heap)
            + len(self._victim_heap) * sys.getsizeof(((), 0, ""))
            + self._entry_metadata_bytes
        )
        stats = {
            "cache_bytes": self.current_bytes,
            "metadata_bytes": metadata_bytes,
            "cache_footprint_bytes": self.current_bytes + metadata_bytes,
            "budget_bytes": self.max_bytes,
            "occupancy": self.current_bytes / self.max_bytes,
            "cached_articles": 0,
            "cached_documents": 0,
            "cached_blocks": len(self.entries),
            "radix_nodes": 0,
            "cached_tokens": self._cached_tokens,
            "useful_tokens": self._cached_tokens,
            "insertions": self.insertions,
            "evictions": self.evictions,
            "policy": self.policy,
            "cache_strategy": self.strategy,
            "useful_bytes": self.current_bytes,
            "shared_bytes": 0,
            "stranded_bytes": 0,
        }
        return stats

    @staticmethod
    def _entry_metadata_size(entry: _FixedEntry) -> int:
        return (
            sys.getsizeof(entry)
            + sys.getsizeof(entry.digest)
            + sys.getsizeof(entry.parent)
            + sys.getsizeof(entry.tokens)
            + len(entry.tokens) * sys.getsizeof(0)
            + sys.getsizeof(entry.payload)
            + sys.getsizeof(entry.payload.blocks)
        )

    def _assert_budget(self):
        assert self.current_bytes <= self.max_bytes
