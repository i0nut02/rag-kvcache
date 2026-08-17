import unittest

from src.quality_cache.data import build_workload
from tests.helpers import articles


class WorkloadTest(unittest.TestCase):
    def test_all_synthetic_traces_are_deterministic(self):
        data = articles()
        for kind in ("random", "zipf"):
            with self.subTest(kind=kind):
                left = build_workload(data, kind, seed=23, requests=100)
                right = build_workload(data, kind, seed=23, requests=100)
                self.assertEqual(
                    [(r.article_id, r.request_id) for r in left],
                    [(r.article_id, r.request_id) for r in right],
                )

    def test_grouped_keeps_article_questions_consecutive(self):
        trace = build_workload(articles(3, 4), "grouped")
        self.assertEqual([r.article_id for r in trace], ["a0"] * 4 + ["a1"] * 4 + ["a2"] * 4)

if __name__ == "__main__":
    unittest.main()
