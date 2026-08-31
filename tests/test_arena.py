from __future__ import annotations

import unittest


try:
    import torch
except ImportError:
    torch = None

from src.quality_cache.caches import CacheKey, DocumentPrefixCache, StoredKV
from src.quality_cache.inference.arena import (
    ArenaCapacityError,
    KVArena,
    StaleArenaHandle,
)


def cache_key(name: str) -> CacheKey:
    return CacheKey(
        name,
        f"hash-{name}",
        "model",
        "tokenizer",
        "prompt",
        "float16",
        "none",
    )


@unittest.skipIf(torch is None, "torch is not installed")
class KVArenaTest(unittest.TestCase):
    def make_cache(self, tokens: int):
        return tuple(
            (
                torch.arange(1 * 2 * tokens * 4, dtype=torch.float16).reshape(
                    1, 2, tokens, 4
                )
                + layer * 100,
                torch.arange(1 * 2 * tokens * 4, dtype=torch.float16).reshape(
                    1, 2, tokens, 4
                )
                + layer * 1000,
            )
            for layer in range(2)
        )

    def make_arena(self, pages: int = 3) -> KVArena:
        # Two layers x K/V x two heads x head-dim four x FP16 = 64 B/token.
        # Four-token pages therefore occupy exactly 256 bytes.
        return KVArena(
            self.make_cache(1),
            pages * 256,
            page_tokens=4,
            device="cpu",
        )

    def test_round_trip_and_page_accounting(self):
        arena = self.make_arena()
        source = self.make_cache(5)
        block = arena.store(source, owner="doc-a")

        self.assertEqual(block.handle.page_indices, (0, 1))
        self.assertEqual(block.stored_bytes, 512)
        self.assertEqual(block.useful_bytes, 320)
        self.assertEqual(block.stranded_bytes, 192)
        self.assertLessEqual(arena.live_allocated_bytes, arena.max_bytes)

        restored = block.restore(dtype=torch.float16, device="cpu")
        for source_layer, restored_layer in zip(source, restored):
            for expected, actual in zip(source_layer, restored_layer):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

        stats = arena.stats()
        self.assertEqual(stats["arena_live_allocations"], 1)
        self.assertEqual(stats["arena_pages_free"], 1)
        self.assertEqual(stats["arena_stranded_bytes"], 192)

    def test_allocation_is_deterministic_after_release(self):
        arena = self.make_arena(pages=4)
        first = arena.store(self.make_cache(5), owner="first")
        second = arena.store(self.make_cache(3), owner="second")
        self.assertEqual(first.handle.page_indices, (0, 1))
        self.assertEqual(second.handle.page_indices, (2,))

        first.release()
        replacement = arena.store(self.make_cache(5), owner="replacement")
        self.assertEqual(replacement.handle.page_indices, (0, 1))

    def test_stale_handle_is_rejected_after_page_reuse(self):
        arena = self.make_arena(pages=2)
        block = arena.store(self.make_cache(5), owner="old")
        stale = block.handle
        block.release()
        replacement = arena.store(self.make_cache(5), owner="new")

        self.assertNotEqual(stale.page_generations, replacement.handle.page_generations)
        with self.assertRaises(StaleArenaHandle):
            arena.restore(stale, dtype=torch.float16, device="cpu")
        with self.assertRaises(StaleArenaHandle):
            arena.release(stale)
        self.assertEqual(arena.stats()["arena_stale_rejections"], 2)

    def test_foreign_handle_is_rejected(self):
        first = self.make_arena()
        second = self.make_arena()
        block = first.store(self.make_cache(2), owner="first")
        with self.assertRaises(StaleArenaHandle):
            second.restore(block.handle, dtype=torch.float16, device="cpu")

    def test_budget_too_small_for_one_page(self):
        with self.assertRaises(ArenaCapacityError):
            KVArena(
                self.make_cache(1),
                255,
                page_tokens=4,
                device="cpu",
            )

    def test_document_reservation_evicts_and_releases_atomically(self):
        l0 = object()
        arena = self.make_arena(pages=2)
        cache = DocumentPrefixCache(512, policy="lru", l0=l0, arena=arena)

        first_block = arena.store(self.make_cache(5), owner="doc-a")
        first_handle = first_block.handle
        self.assertTrue(
            cache.insert(
                cache_key("a"),
                list(range(5)),
                StoredKV(5, blocks=[first_block]),
                1.0,
            )
        )
        incoming_bytes = arena.allocation_bytes_for_tokens(3)
        self.assertTrue(cache.prepare_insert(cache_key("b"), incoming_bytes))
        self.assertEqual(cache.lookup(cache_key("a"), list(range(5))).matched_tokens, 0)
        self.assertEqual(arena.stats()["arena_pages_free"], 2)

        second_block = arena.store(self.make_cache(3), owner="doc-b")
        self.assertTrue(
            cache.insert(
                cache_key("b"),
                list(range(3)),
                StoredKV(3, blocks=[second_block]),
                1.0,
            )
        )
        stats = cache.stats()
        self.assertIs(cache.l0, l0)
        self.assertLessEqual(cache.current_bytes, cache.max_bytes)
        self.assertEqual(stats["cached_documents"], 1)
        self.assertEqual(stats["arena_live_allocations"], 1)
        self.assertEqual(stats["useful_bytes"], 192)
        self.assertEqual(stats["stranded_bytes"], 64)
        match = cache.lookup(cache_key("b"), list(range(3)))
        self.assertEqual(match.stored_bytes, 256)
        self.assertEqual(match.useful_bytes, 192)
        self.assertLessEqual(match.useful_bytes, match.stored_bytes)
        with self.assertRaises(StaleArenaHandle):
            arena.restore(first_handle, dtype=torch.float16, device="cpu")


if __name__ == "__main__":
    unittest.main()
