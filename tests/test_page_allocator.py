from __future__ import annotations

import unittest
from dataclasses import replace

from src.quality_cache.inference.page_allocator import (
    ArenaCapacityError,
    PageAllocator,
    StaleArenaHandle,
)


class PageAllocatorTest(unittest.TestCase):
    def make_allocator(self, pages: int = 4) -> PageAllocator:
        return PageAllocator(
            pages * 256,
            page_tokens=4,
            bytes_per_token=64,
        )

    def test_lowest_free_pages_are_reused_deterministically(self):
        allocator = self.make_allocator()
        first = allocator.allocate(5, owner="first")
        second = allocator.allocate(3, owner="second")
        self.assertEqual(first.page_indices, (0, 1))
        self.assertEqual(second.page_indices, (2,))

        allocator.release(first)
        replacement = allocator.allocate(5, owner="replacement")
        self.assertEqual(replacement.page_indices, (0, 1))
        self.assertNotEqual(first.page_generations, replacement.page_generations)
        allocator.assert_consistent()

    def test_hard_byte_bound_and_capacity_error(self):
        allocator = self.make_allocator(pages=2)
        handle = allocator.allocate(5, owner="full")
        self.assertEqual(allocator.live_allocated_bytes, allocator.reserved_bytes)
        self.assertLessEqual(allocator.live_allocated_bytes, allocator.max_bytes)
        with self.assertRaises(ArenaCapacityError):
            allocator.allocate(1, owner="overflow")
        allocator.release(handle)
        self.assertEqual(allocator.live_allocated_bytes, 0)
        allocator.assert_consistent()

    def test_rejects_stale_foreign_and_malformed_handles(self):
        first = self.make_allocator()
        second = self.make_allocator()
        handle = first.allocate(5, owner="first")
        with self.assertRaises(StaleArenaHandle):
            second.validate(handle)

        malformed = replace(handle, page_indices=(first.page_count,))
        with self.assertRaises(StaleArenaHandle):
            first.validate(malformed)

        first.release(handle)
        with self.assertRaises(StaleArenaHandle):
            first.validate(handle)
        first.assert_consistent()
        second.assert_consistent()

    def test_stats_track_useful_and_stranded_bytes(self):
        allocator = self.make_allocator()
        allocator.allocate(5, owner="doc")
        stats = allocator.stats()
        self.assertEqual(stats["arena_useful_bytes"], 320)
        self.assertEqual(stats["arena_stranded_bytes"], 192)
        self.assertEqual(stats["arena_live_allocations"], 1)
        allocator.assert_consistent()


if __name__ == "__main__":
    unittest.main()
