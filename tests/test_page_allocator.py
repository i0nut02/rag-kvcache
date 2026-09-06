from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

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

    @staticmethod
    def metadata_objects(allocator):
        # Explicit inventory, independent of the implementation's traversal.
        objects = [allocator, vars(allocator)]
        objects.extend(vars(allocator).keys())
        objects.extend(vars(allocator).values())
        objects.extend(allocator._free_pages)
        objects.extend(allocator._page_generations)
        objects.extend(allocator._allocations.keys())
        for handle in allocator._allocations.values():
            objects.extend([handle, vars(handle)])
            objects.extend(vars(handle).keys())
            objects.extend(vars(handle).values())
            objects.extend(handle.page_indices)
            objects.extend(handle.page_generations)
        return {id(obj): obj for obj in objects}

    def test_metadata_includes_nested_objects_once_by_identity(self):
        allocator = self.make_allocator(pages=600)
        owner = "shared-document-owner-" * 20
        first = allocator.allocate(1200, owner=owner)
        second = allocator.allocate(1200, owner=owner)
        self.assertIs(first.owner, second.owner)
        # All pages are allocated: page-index integers remain reachable through
        # handles even though the free-page heap is empty.
        expected = self.metadata_objects(allocator)
        actual = allocator.stats()["arena_metadata_bytes"]
        self.assertEqual(actual, sum(sys.getsizeof(obj) for obj in expected.values()))
        shallow = sum(sys.getsizeof(obj) for obj in (
            allocator._free_pages, allocator._page_generations,
            allocator._allocations, first, second,
        ))
        self.assertGreater(actual, shallow)

        sized = []

        def record_size(obj):
            sized.append(id(obj))
            return 1

        with patch(
            "src.quality_cache.inference.page_allocator.sys.getsizeof",
            side_effect=record_size,
        ):
            self.assertEqual(allocator.stats()["arena_metadata_bytes"], len(expected))
        self.assertCountEqual(sized, expected.keys())
        allocator.assert_consistent()

    def test_metadata_drops_released_handles_and_counts_reused_generations(self):
        allocator = self.make_allocator()
        stale = allocator.allocate(5, owner="old-owner-" * 100)
        allocator.release(stale)
        expected = self.metadata_objects(allocator)
        self.assertNotIn(id(stale), expected)
        self.assertNotIn(id(stale.owner), expected)
        self.assertEqual(
            allocator.stats()["arena_metadata_bytes"],
            sum(sys.getsizeof(obj) for obj in expected.values()),
        )
        # Go past CPython's small-int pool; generation values are real objects
        # shared between the generation table and the new handle.
        for _ in range(300):
            handle = allocator.allocate(5, owner="replacement")
            allocator.release(handle)
        live = allocator.allocate(5, owner="live")
        self.assertGreater(live.page_generations[0], 256)
        expected = self.metadata_objects(allocator)
        before = allocator.stats()
        self.assertEqual(
            before["arena_metadata_bytes"],
            sum(sys.getsizeof(obj) for obj in expected.values()),
        )
        self.assertEqual(allocator.stats(), before)
        allocator.assert_consistent()


if __name__ == "__main__":
    unittest.main()
