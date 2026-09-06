from __future__ import annotations

import gc
import unittest
import weakref
from unittest.mock import patch

from src.quality_cache.caches import RadixPrefixCache, StoredKV
from src.quality_cache.caches.base import cache_namespace
from tests.test_cache_strategies import key


POLICIES = ("lru", "lfu", "fifo", "gdsf")


class RadixAccessTest(unittest.TestCase):
    def test_admission_is_not_an_extra_lookup(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                cache = RadixPrefixCache(40, policy=policy)
                tokens = [1, 2]
                self.assertEqual(cache.lookup(key("a"), tokens).matched_tokens, 0)
                with patch.object(cache, "lookup", side_effect=AssertionError("extra lookup")):
                    self.assertTrue(cache.insert(key("a"), tokens, StoredKV(2, simulated_bytes=20), 2.0))
                node = cache.roots[cache_namespace(key("a"))].children[1]
                self.assertEqual(cache.clock, 2)
                self.assertEqual(node.frequency, 1)
                self.assertEqual(node.last_access, 2)
                self.assertAlmostEqual(node.priority, 0.1)
                self.assertEqual(cache.lookup(key("a"), tokens).matched_tokens, 2)
                self.assertEqual(node.frequency, 2)
                self.assertEqual(node.last_access, 3)
                if policy == "gdsf":
                    self.assertAlmostEqual(node.priority, 0.2)

    def test_partial_lookup_then_insert_counts_shared_prefix_once(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                cache = RadixPrefixCache(60, policy=policy)
                cache.insert(key("a"), [1, 2, 3, 4], StoredKV(4, simulated_bytes=40), 1.0)
                match = cache.lookup(key("b"), [1, 2, 9, 10])
                self.assertEqual(match.matched_tokens, 2)
                self.assertEqual(match.hit_ratio, 0.5)
                cache.insert(key("b"), [1, 2, 9, 10], StoredKV(4, simulated_bytes=40), 1.0)
                shared = cache.roots[cache_namespace(key("a"))].children[1]
                self.assertEqual(shared.frequency, 2)
                self.assertEqual(shared.children[3].frequency, 2)
                self.assertEqual(shared.children[9].frequency, 1)
                self.assertEqual(cache.current_bytes, 60)
                cache.lookup(key("b"), [1, 2, 9, 10])
                self.assertEqual(shared.frequency, 3)
                self.assertEqual(shared.children[9].frequency, 2)
                self.assertEqual(shared.children[3].frequency, 2)

    def test_reinsertion_refreshes_recency_without_incrementing_frequency(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                cache = RadixPrefixCache(40, policy=policy)
                cache.insert(key("a"), [1, 2], StoredKV(2, simulated_bytes=20), 1.0)
                cache.insert(key("b"), [3, 4], StoredKV(2, simulated_bytes=20), 1.0)
                cache.insert(key("a"), [1, 2], StoredKV(2, simulated_bytes=20), 1.0)
                root = cache.roots[cache_namespace(key("a"))]
                self.assertEqual(root.children[1].frequency, 1)
                self.assertEqual(root.children[1].last_access, cache.clock)
                # The current heap candidate must agree with a direct scan,
                # including after insertion updates an already resident leaf.
                expected = min(cache._nodes_by_id.values(), key=cache._victim_priority)
                cache.insert(key("c"), [5, 6], StoredKV(2, simulated_bytes=20), 1.0)
                self.assertNotIn(expected.node_id, cache._nodes_by_id)
                self.assertIn(5, root.children)
                self.assertEqual(set(root.children), {3, 5} if policy == "fifo" else {1, 5})

    def test_admission_reports_remaining_prefix_after_budget_evictions(self):
        for policy in POLICIES:
            with self.subTest(policy=policy):
                cache = RadixPrefixCache(20, policy=policy)
                self.assertFalse(cache.insert(key("a"), [], StoredKV(0), 1.0))
                self.assertEqual(cache.clock, 0)
                # A single oversized edge is completely evicted.
                self.assertFalse(cache.insert(key("a"), [1, 2, 3], StoredKV(3, simulated_bytes=30), 1.0))
                self.assertEqual(cache.current_bytes, 0)
                self.assertTrue(cache.insert(key("a"), [1, 2], StoredKV(2, simulated_bytes=20), 1.0))
                # The new tail cannot fit, but the shared prefix remains.
                self.assertTrue(cache.insert(key("b"), [1, 2, 3], StoredKV(3, simulated_bytes=30), 1.0))
                self.assertEqual(cache.lookup(key("b"), [1, 2, 3]).matched_tokens, 2)
                self.assertLessEqual(cache.current_bytes, cache.max_bytes)

    def test_eviction_releases_node_and_payload_but_preserves_l0(self):
        l0 = object()
        cache = RadixPrefixCache(20, l0=l0)
        cache.insert(key("a"), [1, 2], StoredKV(2, simulated_bytes=20), 1.0)
        node = next(iter(cache._nodes_by_id.values()))
        node_ref, payload_ref = weakref.ref(node), weakref.ref(node.payload)
        del node
        cache.insert(key("b"), [3, 4], StoredKV(2, simulated_bytes=20), 1.0)
        gc.collect()
        self.assertIsNone(node_ref())
        self.assertIsNone(payload_ref())
        self.assertIs(cache.l0, l0)
        self.assertEqual(cache.current_bytes, 20)


if __name__ == "__main__":
    unittest.main()
