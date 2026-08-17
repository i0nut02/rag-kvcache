from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from ..inference.tensors import KVBlock


POLICIES = ("lru", "lfu", "fifo", "gdsf")


@dataclass(frozen=True)
class CacheKey:
    article_id: str
    article_hash: str
    model_revision: str
    tokenizer_revision: str
    prompt_version: str
    dtype: str
    quantization: str


@dataclass
class CacheEntry:
    key: CacheKey
    token_count: int
    prefill_cost_s: float
    blocks: list[KVBlock] = field(default_factory=list)
    simulated_bytes: int | None = None
    frequency: int = 1
    inserted_at: int = 0
    last_access: int = 0
    priority: float = 0.0

    @property
    def stored_bytes(self) -> int:
        if self.simulated_bytes is not None:
            return int(self.simulated_bytes)
        return sum(block.stored_bytes for block in self.blocks)


class BudgetTooSmall(ValueError):
    pass


class ArticleKVCache:
    """Byte-bounded article cache. L0 is pinned separately and never evicted."""

    def __init__(
        self,
        max_bytes: int,
        *,
        max_articles: int | None = None,
        policy: str = "lru",
        minimum_article_bytes: int | None = None,
        l0: Any = None,
    ):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if max_articles is not None and max_articles <= 0:
            raise ValueError("max_articles must be positive")
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if minimum_article_bytes is not None and max_bytes < minimum_article_bytes:
            raise BudgetTooSmall("cache budget cannot hold the smallest article")
        self.max_bytes = int(max_bytes)
        self.max_articles = max_articles
        self.policy = policy
        self.l0 = l0
        self.entries: dict[CacheKey, CacheEntry] = {}
        self.current_bytes = 0
        self.clock = 0
        self.gdsf_clock = 0.0
        self.insertions = 0
        self.evictions = 0
        self._stats_cache: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, key: CacheKey) -> CacheEntry | None:
        self.clock += 1
        entry = self.entries.get(key)
        if entry is None:
            return None
        entry.frequency += 1
        entry.last_access = self.clock
        if self.policy == "gdsf":
            entry.priority = self._gdsf_priority(entry)
        return entry

    def put(self, entry: CacheEntry) -> list[CacheKey]:
        size = entry.stored_bytes
        if size > self.max_bytes:
            raise BudgetTooSmall(
                f"article requires {size} bytes but cache budget is {self.max_bytes}"
            )
        self.clock += 1
        old = self.entries.pop(entry.key, None)
        if old is not None:
            self.current_bytes -= old.stored_bytes
        entry.inserted_at = self.clock
        entry.last_access = self.clock
        entry.frequency = max(1, entry.frequency)
        entry.priority = self._gdsf_priority(entry)
        evicted: list[CacheKey] = []
        while self._would_exceed(size):
            victim = self._victim()
            if victim is None:
                raise RuntimeError("cache could not select an eviction victim")
            evicted.append(victim.key)
            self._remove(victim)
        self.entries[entry.key] = entry
        self.current_bytes += size
        self.insertions += 1
        self._invalidate_stats()
        self._assert_budget()
        return evicted

    def prefill(self, entries: list[CacheEntry]) -> list[CacheKey]:
        evicted = []
        for entry in entries:
            evicted.extend(self.put(entry))
        return evicted

    def _would_exceed(self, incoming_bytes: int) -> bool:
        byte_over = self.current_bytes + incoming_bytes > self.max_bytes
        count_over = self.max_articles is not None and len(self.entries) + 1 > self.max_articles
        return byte_over or count_over

    def _victim(self) -> CacheEntry | None:
        entries = list(self.entries.values())
        if not entries:
            return None
        if self.policy == "lru":
            return min(entries, key=lambda item: (item.last_access, item.inserted_at))
        if self.policy == "lfu":
            return min(entries, key=lambda item: (item.frequency, item.last_access))
        if self.policy == "fifo":
            return min(entries, key=lambda item: item.inserted_at)
        return min(entries, key=lambda item: (item.priority, item.last_access))

    def _gdsf_priority(self, entry: CacheEntry) -> float:
        return self.gdsf_clock + entry.frequency * max(entry.prefill_cost_s, 1e-12) / max(entry.stored_bytes, 1)

    def _remove(self, entry: CacheEntry) -> None:
        if self.policy == "gdsf":
            self.gdsf_clock = entry.priority
        removed = self.entries.pop(entry.key)
        self.current_bytes -= removed.stored_bytes
        removed.blocks.clear()  # make evicted tensors unreachable through the entry
        self.evictions += 1
        self._invalidate_stats()

    def _assert_budget(self) -> None:
        assert self.current_bytes <= self.max_bytes
        if self.max_articles is not None:
            assert len(self.entries) <= self.max_articles

    def stats(self) -> dict[str, Any]:
        if self._stats_cache is not None:
            return dict(self._stats_cache)
        metadata_bytes = sys.getsizeof(self.entries)
        for key, entry in self.entries.items():
            metadata_bytes += sys.getsizeof(key) + sys.getsizeof(entry)
            metadata_bytes += sum(sys.getsizeof(value) for value in key.__dict__.values())
            metadata_bytes += sys.getsizeof(entry.blocks)
        stats = {
            "cache_bytes": self.current_bytes,
            "metadata_bytes": metadata_bytes,
            "cache_footprint_bytes": self.current_bytes + metadata_bytes,
            "budget_bytes": self.max_bytes,
            "occupancy": self.current_bytes / self.max_bytes,
            "cached_articles": len(self.entries),
            "cached_tokens": sum(entry.token_count for entry in self.entries.values()),
            "insertions": self.insertions,
            "evictions": self.evictions,
            "policy": self.policy,
        }
        self._stats_cache = dict(stats)
        return stats

    def _invalidate_stats(self) -> None:
        self._stats_cache = None
