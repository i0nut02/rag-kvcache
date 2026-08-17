import time
import unittest

from src.quality_cache.data import build_workload
from src.quality_cache.reporting import summarize
from src.quality_cache.simulation import article_sizes, simulate_trace, working_set_bytes
from tests.helpers import articles


class SimulatorTest(unittest.TestCase):
    def test_grouped_maximum_is_first_miss_per_article(self):
        trace = build_workload(articles(5, 4), "grouped")
        sizes, _ = article_sizes(trace, "Qwen/Qwen2.5-0.5B-Instruct", "cpu-fp16")
        rows = simulate_trace(trace, policy="lru", budget_bytes=working_set_bytes(sizes))
        summary = summarize(rows)
        self.assertAlmostEqual(summary["full_document_hit_rate"], 15 / 20)
        self.assertGreater(summary["request_hit_rate"], 0.0)
        self.assertLess(summary["request_hit_rate"], 1.0)
        self.assertGreater(summary["prefix_token_hit_rate"], 0.0)
        self.assertLess(summary["prefix_token_hit_rate"], 1.0)
        self.assertGreater(summary["article_token_hit_rate"], 0.0)
        self.assertLess(summary["article_token_hit_rate"], 1.0)
        self.assertEqual(summary["avoided_prefill_tokens"], summary["matched_prefill_tokens"])

    def test_twenty_thousand_request_no_inference_target(self):
        trace = build_workload(articles(40, 4), "zipf", requests=20_000, seed=42)
        sizes, _ = article_sizes(trace, "Qwen/Qwen2.5-0.5B-Instruct", "cpu-fp16")
        started = time.perf_counter()
        rows = simulate_trace(
            trace,
            policy="gdsf",
            budget_bytes=max(min(sizes.values()), working_set_bytes(sizes) // 10),
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(len(rows), 20_000)
        self.assertLess(elapsed, 60.0)
        self.assertTrue(all(row["cache_bytes"] <= row["budget_bytes"] for row in rows))

    def test_articles_larger_than_budget_are_uncached_misses(self):
        trace = build_workload(articles(2, 2), "grouped")
        sizes, _ = article_sizes(trace, "Qwen/Qwen2.5-0.5B-Instruct", "cpu-fp16")
        rows = simulate_trace(trace, policy="lru", budget_bytes=sizes["a0"])
        oversized_rows = [row for row in rows if row["article_id"] == "a1"]
        self.assertEqual([row["cache_hit"] for row in oversized_rows], [False, False])
        self.assertTrue(all(row["cache_bytes"] <= row["budget_bytes"] for row in rows))


if __name__ == "__main__":
    unittest.main()
