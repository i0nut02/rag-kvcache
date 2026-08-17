"""Whole-document cache strategy with a pinned L0 root."""

from __future__ import annotations

from typing import Any

from .article import ArticleKVCache, BudgetTooSmall, CacheEntry, CacheKey
from .base import PrefixLookup, StoredKV


class DocumentPrefixCache:
    """One-level prefix tree: pinned L0 root with atomic document children."""

    strategy = "document"

    def __init__(self, max_bytes: int, *, policy: str = "lru", max_articles=None, l0=None):
        self.cache = ArticleKVCache(
            max_bytes, policy=policy, max_articles=max_articles, l0=l0
        )

    @property
    def l0(self):
        return self.cache.l0

    @property
    def max_bytes(self):
        return self.cache.max_bytes

    @property
    def current_bytes(self):
        return self.cache.current_bytes

    def __len__(self):
        return len(self.cache)

    def lookup(self, key: CacheKey, tokens: list[int]) -> PrefixLookup:
        entry = self.cache.get(key)
        if entry is None or entry.token_count != len(tokens):
            return PrefixLookup(requested_tokens=len(tokens))
        payload = StoredKV(
            entry.token_count,
            blocks=entry.blocks,
            simulated_bytes=entry.simulated_bytes,
        )
        return PrefixLookup(entry.token_count, [payload], len(tokens))

    def insert(
        self, key: CacheKey, tokens: list[int], payload: StoredKV, prefill_cost_s: float
    ) -> bool:
        if payload.stored_bytes > self.max_bytes:
            return False
        # Document mode is deliberately one physical object, even if its source
        # tensors arrived as smaller temporary blocks.
        entry = CacheEntry(
            key=key,
            token_count=len(tokens),
            prefill_cost_s=prefill_cost_s,
            blocks=list(payload.blocks),
            simulated_bytes=payload.simulated_bytes,
        )
        try:
            self.cache.put(entry)
        except BudgetTooSmall:
            return False
        return True

    def stats(self) -> dict[str, Any]:
        stats = self.cache.stats()
        stats.update(
            {
                "cache_strategy": self.strategy,
                "root_nodes": 1 if self.l0 is not None else 0,
                "document_tree_nodes": len(self.cache) + (1 if self.l0 is not None else 0),
                "cached_documents": len(self.cache),
                "cached_blocks": 0,
                "radix_nodes": 0,
                "cached_tokens": sum(
                    entry.token_count for entry in self.cache.entries.values()
                ),
                "useful_bytes": self.current_bytes,
                "shared_bytes": 0,
                "stranded_bytes": 0,
            }
        )
        return stats
