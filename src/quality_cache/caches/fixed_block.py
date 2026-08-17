"""Content-addressed fixed-size prefix-block cache strategy."""

from __future__ import annotations

import hashlib
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
        self.current_bytes = 0
        self.clock = 0
        self.gdsf_clock = 0.0
        self.insertions = 0
        self.evictions = 0
        self._stats_cache: dict[str, Any] | None = None

    def __len__(self):
        return len(self.entries)

    def _root(self, key: CacheKey) -> str:
        root = cache_namespace(key)
        if root not in self.root_hashes:
            self.root_hashes.add(root)
            self._invalidate_stats()
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
            self.entries[digest] = entry
            self.current_bytes += size
            self.insertions += 1
            self._invalidate_stats()
            admitted = True
            protected.add(digest)
            parent = digest
        self._assert_budget()
        return admitted

    def _evict_one(self, protected: set[str] | None = None) -> bool:
        protected = protected or set()
        parent_digests = {entry.parent for entry in self.entries.values()}
        leaves = [
            entry
            for digest, entry in self.entries.items()
            if digest not in protected and digest not in parent_digests
        ]
        if not leaves:
            return False
        # Evict complete-prefix leaves first so retained descendants never lose
        # their parent chain and become unusable cache bytes.
        victim = self._victim(leaves)
        self.entries.pop(victim.digest)
        self.current_bytes -= victim.payload.stored_bytes
        if self.policy == "gdsf":
            self.gdsf_clock = victim.priority
        victim.payload.blocks.clear()
        self.evictions += 1
        self._invalidate_stats()
        return True

    def _victim(self, entries: list[_FixedEntry]) -> _FixedEntry:
        if self.policy == "lru":
            return min(entries, key=lambda entry: (entry.last_access, entry.inserted_at))
        if self.policy == "lfu":
            return min(
                entries,
                key=lambda entry: (entry.frequency, entry.last_access, entry.inserted_at),
            )
        if self.policy == "fifo":
            return min(entries, key=lambda entry: entry.inserted_at)
        return min(
            entries,
            key=lambda entry: (entry.priority, entry.last_access, entry.inserted_at),
        )

    def _gdsf_priority(self, entry: _FixedEntry) -> float:
        return (
            self.gdsf_clock
            + entry.frequency
            * max(entry.prefill_cost_s, 1e-12)
            / max(entry.payload.stored_bytes, 1)
        )

    def stats(self) -> dict[str, Any]:
        if self._stats_cache is not None:
            return dict(self._stats_cache)
        children: dict[str, list[str]] = {}
        for digest, entry in self.entries.items():
            children.setdefault(entry.parent, []).append(digest)
        reachable = set()
        frontier = list(self.root_hashes)
        while frontier:
            parent = frontier.pop()
            for digest in children.get(parent, ()):
                if digest not in reachable:
                    reachable.add(digest)
                    frontier.append(digest)
        useful = sum(self.entries[digest].payload.stored_bytes for digest in reachable)
        stranded = self.current_bytes - useful
        metadata_bytes = sys.getsizeof(self.entries) + sys.getsizeof(self.root_hashes)
        for entry in self.entries.values():
            metadata_bytes += sys.getsizeof(entry)
            metadata_bytes += sys.getsizeof(entry.digest) + sys.getsizeof(entry.parent)
            metadata_bytes += sys.getsizeof(entry.tokens)
            metadata_bytes += len(entry.tokens) * sys.getsizeof(0)
            metadata_bytes += sys.getsizeof(entry.payload)
            metadata_bytes += sys.getsizeof(entry.payload.blocks)
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
            "cached_tokens": sum(entry.token_count for entry in self.entries.values()),
            "useful_tokens": sum(
                self.entries[digest].token_count for digest in reachable
            ),
            "insertions": self.insertions,
            "evictions": self.evictions,
            "policy": self.policy,
            "cache_strategy": self.strategy,
            "useful_bytes": useful,
            "shared_bytes": 0,
            "stranded_bytes": stranded,
        }
        self._stats_cache = dict(stats)
        return stats

    def _assert_budget(self):
        assert self.current_bytes <= self.max_bytes

    def _invalidate_stats(self) -> None:
        self._stats_cache = None
