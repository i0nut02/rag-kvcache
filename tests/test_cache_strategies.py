from __future__ import annotations

import unittest

from src.quality_cache.caches import (
    CacheKey,
    DocumentPrefixCache,
    FixedBlockPrefixCache,
    RadixPrefixCache,
    StoredKV,
)


def key(name: str, article_hash: str | None = None) -> CacheKey:
    return CacheKey(
        name,
        article_hash or f"hash-{name}",
        "model",
        "tokenizer",
        "prompt",
        "float16",
        "none",
    )


class CacheStrategiesTest(unittest.TestCase):
    def test_document_cache_is_atomic_and_keeps_l0_pinned(self):
        l0 = object()
        cache = DocumentPrefixCache(60, policy="lru", l0=l0)
        self.assertTrue(cache.insert(key("a"), [1, 2, 3], StoredKV(3, simulated_bytes=60), 1.0))
        self.assertEqual(cache.lookup(key("a"), [1, 2, 3]).matched_tokens, 3)
        self.assertTrue(cache.insert(key("b"), [4, 5], StoredKV(2, simulated_bytes=50), 1.0))
        self.assertEqual(cache.lookup(key("a"), [1, 2, 3]).matched_tokens, 0)
        self.assertEqual(cache.lookup(key("b"), [4, 5]).matched_tokens, 2)
        self.assertIs(cache.l0, l0)
        self.assertLessEqual(cache.current_bytes, cache.max_bytes)
        self.assertEqual(cache.stats()["cached_documents"], 1)
        self.assertEqual(cache.stats()["root_nodes"], 1)
        self.assertEqual(cache.stats()["document_tree_nodes"], 2)

    def test_document_larger_than_budget_remains_uncached(self):
        cache = DocumentPrefixCache(10)
        self.assertFalse(
            cache.insert(key("a"), [1, 2], StoredKV(2, simulated_bytes=11), 1.0)
        )
        self.assertEqual(cache.lookup(key("a"), [1, 2]).matched_tokens, 0)

    def test_fixed_blocks_restore_only_complete_chain(self):
        cache = FixedBlockPrefixCache(100, block_tokens=4)
        tokens = list(range(10))
        cache.insert(key("a"), tokens, StoredKV(10, simulated_bytes=100), 1.0)
        match = cache.lookup(key("a"), tokens)
        self.assertEqual(match.matched_tokens, 8)
        self.assertEqual(match.requested_tokens, 10)
        self.assertAlmostEqual(match.hit_ratio, 0.8)
        self.assertEqual(match.stored_bytes, 80)
        self.assertEqual(cache.stats()["cached_blocks"], 2)
        self.assertEqual(cache.stats()["cached_tokens"], 8)
        self.assertLessEqual(cache.current_bytes, cache.max_bytes)

    def test_fixed_block_budget_keeps_a_reachable_prefix(self):
        cache = FixedBlockPrefixCache(40, block_tokens=4)
        tokens = list(range(8))
        cache.insert(key("a"), tokens, StoredKV(8, simulated_bytes=80), 1.0)
        self.assertEqual(cache.lookup(key("a"), tokens).matched_tokens, 4)
        self.assertEqual(cache.stats()["stranded_bytes"], 0)
        self.assertEqual(cache.stats()["useful_bytes"], 40)
        self.assertLessEqual(cache.current_bytes, cache.max_bytes)

    def test_fixed_block_cost_is_proportional_to_total_article_tokens(self):
        cache = FixedBlockPrefixCache(100, block_tokens=4, policy="gdsf")
        tokens = list(range(10))
        cache.insert(key("a"), tokens, StoredKV(10, simulated_bytes=100), 2.0)
        self.assertEqual(len(cache.entries), 2)
        self.assertAlmostEqual(
            sum(entry.prefill_cost_s for entry in cache.entries.values()), 1.6
        )
        self.assertGreater(cache.stats()["metadata_bytes"], 0)
        self.assertGreater(cache.stats()["cache_footprint_bytes"], cache.current_bytes)

    def test_radix_shares_prefix_and_returns_longest_match(self):
        cache = RadixPrefixCache(100)
        cache.insert(key("a"), [1, 2, 3, 4], StoredKV(4, simulated_bytes=40), 1.0)
        cache.insert(key("b"), [1, 2, 9, 10], StoredKV(4, simulated_bytes=40), 1.0)
        self.assertEqual(cache.lookup(key("a"), [1, 2, 3, 4]).matched_tokens, 4)
        self.assertEqual(cache.lookup(key("b"), [1, 2, 9, 10]).matched_tokens, 4)
        partial = cache.lookup(key("c"), [1, 2, 7])
        self.assertEqual(partial.matched_tokens, 2)
        self.assertAlmostEqual(partial.hit_ratio, 2 / 3)
        stats = cache.stats()
        self.assertEqual(stats["shared_bytes"], 20)
        self.assertEqual(stats["radix_nodes"], 3)
        self.assertEqual(stats["cached_tokens"], 6)
        self.assertTrue(
            all(
                node.token_count == len(node.tokens) == node.payload.token_count
                for node in cache._nodes()
            )
        )
        self.assertLessEqual(cache.current_bytes, cache.max_bytes)

    def test_radix_evicts_cold_leaf_and_preserves_shared_prefix(self):
        cache = RadixPrefixCache(45)
        cache.insert(key("a"), [1, 2, 3, 4], StoredKV(4, simulated_bytes=40), 1.0)
        cache.insert(key("b"), [1, 2, 9, 10], StoredKV(4, simulated_bytes=40), 1.0)
        self.assertEqual(cache.lookup(key("a"), [1, 2, 3, 4]).matched_tokens, 2)
        self.assertEqual(cache.lookup(key("b"), [1, 2, 9, 10]).matched_tokens, 4)
        self.assertLessEqual(cache.current_bytes, cache.max_bytes)

    def test_radix_lfu_keeps_frequently_used_leaf(self):
        cache = RadixPrefixCache(40, policy="lfu")
        cache.insert(key("a"), [1, 2], StoredKV(2, simulated_bytes=20), 1.0)
        cache.insert(key("b"), [3, 4], StoredKV(2, simulated_bytes=20), 1.0)
        cache.lookup(key("a"), [1, 2])
        cache.insert(key("c"), [5, 6], StoredKV(2, simulated_bytes=20), 1.0)
        self.assertEqual(cache.lookup(key("a"), [1, 2]).matched_tokens, 2)
        self.assertEqual(cache.lookup(key("b"), [3, 4]).matched_tokens, 0)
        self.assertEqual(cache.lookup(key("c"), [5, 6]).matched_tokens, 2)
        self.assertEqual(cache.stats()["policy"], "lfu")

    def test_radix_gdsf_prefers_high_cost_per_byte_leaf(self):
        cache = RadixPrefixCache(40, policy="gdsf")
        cache.insert(key("a"), [1, 2], StoredKV(2, simulated_bytes=20), 100.0)
        cache.insert(key("b"), [3, 4], StoredKV(2, simulated_bytes=20), 1.0)
        cache.insert(key("c"), [5, 6], StoredKV(2, simulated_bytes=20), 3.0)
        self.assertEqual(cache.lookup(key("a"), [1, 2]).matched_tokens, 2)
        self.assertEqual(cache.lookup(key("b"), [3, 4]).matched_tokens, 0)
        self.assertEqual(cache.lookup(key("c"), [5, 6]).matched_tokens, 2)
        self.assertEqual(cache.stats()["policy"], "gdsf")

    def test_namespace_identity_prevents_cross_model_hits(self):
        cache = RadixPrefixCache(100)
        original = key("a")
        changed = CacheKey(
            "a", original.article_hash, "other-model", "tokenizer", "prompt", "float16", "none"
        )
        cache.insert(original, [1, 2, 3], StoredKV(3, simulated_bytes=30), 1.0)
        self.assertEqual(cache.lookup(changed, [1, 2, 3]).matched_tokens, 0)


if __name__ == "__main__":
    unittest.main()
