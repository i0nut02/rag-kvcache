import unittest

from src.quality_cache.reporting import summarize


class MetricsTest(unittest.TestCase):
    def test_fractional_token_hit_rates_are_macro_and_micro_averaged(self):
        rows = [
            {
                "cache_hit": True,
                "article_id": "a",
                "partial_cache_hit": True,
                "partial_article_hit": False,
                "partial_article_text_hit": False,
                "partial_document_tree_hit": False,
                "root_only_hit": False,
                "cache_hit_ratio": 0.8,
                "article_text_cache_hit_ratio": 1.0,
                "document_tree_hit_ratio": 1.0,
                "cached_prompt_tokens": 80,
                "total_prompt_tokens": 100,
                "document_tree_cached_tokens": 70,
                "document_tree_total_tokens": 70,
                "matched_prefix_tokens": 70,
                "article_tokens": 70,
                "matched_cache_bytes": 8,
                "article_bytes": 10,
                "ttft_s": -1,
            },
            {
                "cache_hit": False,
                "article_id": "b",
                "partial_cache_hit": True,
                "partial_article_hit": True,
                "partial_article_text_hit": True,
                "partial_document_tree_hit": True,
                "root_only_hit": False,
                "cache_hit_ratio": 0.2,
                "article_text_cache_hit_ratio": 0.2,
                "document_tree_hit_ratio": 0.25,
                "cached_prompt_tokens": 10,
                "total_prompt_tokens": 50,
                "document_tree_cached_tokens": 5,
                "document_tree_total_tokens": 20,
                "matched_prefix_tokens": 4,
                "article_tokens": 20,
                "matched_cache_bytes": 2,
                "article_bytes": 10,
                "ttft_s": -1,
            },
        ]
        summary = summarize(rows, cold_requests=1)
        self.assertAlmostEqual(summary["request_hit_rate"], 0.5)
        self.assertEqual(summary["distinct_articles"], 2)
        self.assertAlmostEqual(summary["prefix_token_hit_rate"], 0.6)
        self.assertAlmostEqual(summary["byte_hit_rate"], 0.5)
        self.assertAlmostEqual(summary["full_document_hit_rate"], 0.5)
        self.assertAlmostEqual(summary["partial_prefix_hit_rate"], 1.0)
        self.assertAlmostEqual(summary["partial_article_hit_rate"], 0.5)
        self.assertAlmostEqual(summary["partial_article_text_hit_rate"], 0.5)
        self.assertAlmostEqual(summary["article_text_hit_rate"], 0.6)
        self.assertAlmostEqual(summary["document_tree_hit_rate"], 0.625)
        self.assertAlmostEqual(summary["document_tree_token_hit_rate"], 75 / 90)
        self.assertAlmostEqual(summary["partial_document_tree_hit_rate"], 0.5)
        self.assertAlmostEqual(summary["root_only_hit_rate"], 0.0)
        self.assertAlmostEqual(summary["article_token_hit_rate"], 74 / 90)
        self.assertEqual(summary["requested_article_tokens"], 90)
        self.assertAlmostEqual(summary["cold_hit_rate"], 0.8)
        self.assertAlmostEqual(summary["steady_state_hit_rate"], 0.2)

    def test_reference_agreement_and_accuracy_delta(self):
        rows = [
            {
                "cache_hit": True,
                "article_bytes": 10,
                "ttft_s": 0.1,
                "predicted_label": "A",
                "gold_label": "A",
                "fp16_reference_label": "A",
                "reference_agreement": True,
            },
            {
                "cache_hit": False,
                "article_bytes": 10,
                "ttft_s": 0.2,
                "predicted_label": "B",
                "gold_label": "A",
                "fp16_reference_label": "A",
                "reference_agreement": False,
            },
        ]
        summary = summarize(rows, cold_requests=1)
        self.assertEqual(summary["reference_label_agreement"], 0.5)
        self.assertEqual(summary["fp16_reference_accuracy"], 1.0)
        self.assertEqual(summary["accuracy_delta_vs_fp16"], -0.5)
        self.assertEqual(summary["cold_hit_rate"], 1.0)
        self.assertEqual(summary["steady_state_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
