from __future__ import annotations

import unittest

from src.quality_cache.caches import ArticleKVCache, BudgetTooSmall, CacheEntry, CacheKey


class FakeBlock:
    def __init__(self, size):
        self.stored_bytes = size


def key(name):
    return CacheKey(name, "hash", "model", "tokenizer", "prompt", "float16", "none")


def entry(name, size=1, cost=1.0):
    return CacheEntry(key(name), 10, cost, blocks=[FakeBlock(size)])


class CachePolicyTest(unittest.TestCase):
    def test_lru_lfu_fifo_and_gdsf_known_outcomes(self):
        expected_victim = {"lru": "B", "lfu": "B", "fifo": "A", "gdsf": "B"}
        for policy, victim in expected_victim.items():
            with self.subTest(policy=policy):
                cache = ArticleKVCache(2, policy=policy, l0=object())
                cache.put(entry("A"))
                cache.put(entry("B"))
                cache.get(key("A"))
                cache.put(entry("C"))
                self.assertNotIn(key(victim), cache.entries)
                self.assertLessEqual(cache.current_bytes, cache.max_bytes)

    def test_lfu_uses_lru_tie_breaking(self):
        cache = ArticleKVCache(2, policy="lfu")
        cache.put(entry("A"))
        cache.put(entry("B"))
        cache.put(entry("C"))
        self.assertNotIn(key("A"), cache.entries)

    def test_budget_and_article_count_hold_after_every_put(self):
        cache = ArticleKVCache(7, max_articles=2, policy="lru")
        for name, size in zip("ABCDE", [3, 4, 2, 5, 1]):
            cache.put(entry(name, size))
            self.assertLessEqual(cache.current_bytes, 7)
            self.assertLessEqual(len(cache), 2)

    def test_oversized_article_is_rejected(self):
        cache = ArticleKVCache(4)
        with self.assertRaises(BudgetTooSmall):
            cache.put(entry("A", 5))
        with self.assertRaises(BudgetTooSmall):
            ArticleKVCache(4, minimum_article_bytes=5)

    def test_evicted_blocks_are_unreachable_and_l0_remains_pinned(self):
        l0 = object()
        cache = ArticleKVCache(1, l0=l0)
        victim = entry("A")
        cache.put(victim)
        cache.put(entry("B"))
        self.assertEqual(victim.blocks, [])
        self.assertIs(cache.l0, l0)
        self.assertNotIn(key("A"), cache.entries)

    def test_full_identity_participates_in_lookup(self):
        cache = ArticleKVCache(2)
        cache.put(entry("A"))
        changed = CacheKey("A", "changed", "model", "tokenizer", "prompt", "float16", "none")
        self.assertIsNone(cache.get(changed))


if __name__ == "__main__":
    unittest.main()
