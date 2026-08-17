import unittest

from src.quality_cache.caches.offline import simulate_farthest_next_use


class FarthestNextUseTest(unittest.TestCase):
    def test_known_trace(self):
        result = simulate_farthest_next_use(
            list("ABCAB"), {letter: 1 for letter in "ABC"}, 2
        )
        self.assertEqual(result.hits, (False, False, False, True, False))
        self.assertEqual(result.final_bytes, 2)
        self.assertTrue(result.optimal_for_equal_sizes)

    def test_variable_sizes_never_exceed_budget(self):
        result = simulate_farthest_next_use(
            list("ABACBC"), {"A": 3, "B": 2, "C": 4}, 5
        )
        self.assertLessEqual(result.final_bytes, 5)
        self.assertFalse(result.optimal_for_equal_sizes)


if __name__ == "__main__":
    unittest.main()
