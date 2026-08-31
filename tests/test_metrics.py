import unittest

from src.quality_cache.reporting import summarize


class MetricsTest(unittest.TestCase):
    def test_arena_capacity_and_fragmentation_metrics_are_aggregated(self):
        rows = [
            {
                "cache_hit": False,
                "ttft_s": -1,
                "arena_reserved_bytes": 4096,
                "arena_free_bytes": 3072,
                "arena_peak_allocated_bytes": 1024,
                "arena_metadata_bytes": 128,
                "arena_pages_total": 4,
                "arena_pages_free": 3,
                "arena_live_allocations": 1,
                "arena_allocations": 1,
                "arena_releases": 0,
                "arena_stale_rejections": 0,
                "arena_useful_bytes": 768,
                "arena_stranded_bytes": 256,
            },
            {
                "cache_hit": True,
                "ttft_s": -1,
                "arena_reserved_bytes": 4096,
                "arena_free_bytes": 1024,
                "arena_peak_allocated_bytes": 3072,
                "arena_metadata_bytes": 192,
                "arena_pages_total": 4,
                "arena_pages_free": 1,
                "arena_live_allocations": 3,
                "arena_allocations": 4,
                "arena_releases": 1,
                "arena_stale_rejections": 2,
                "arena_useful_bytes": 2500,
                "arena_stranded_bytes": 572,
            },
        ]

        summary = summarize(rows)

        self.assertEqual(summary["arena_reserved_bytes_peak"], 4096)
        self.assertEqual(summary["arena_free_bytes_min"], 1024)
        self.assertEqual(summary["arena_peak_allocated_bytes"], 3072)
        self.assertEqual(summary["arena_metadata_bytes_peak"], 192)
        self.assertEqual(summary["arena_pages_total_peak"], 4)
        self.assertEqual(summary["arena_pages_free_min"], 1)
        self.assertEqual(summary["arena_live_allocations_peak"], 3)
        self.assertEqual(summary["arena_allocations"], 4)
        self.assertEqual(summary["arena_releases"], 1)
        self.assertEqual(summary["arena_stale_rejections"], 2)
        self.assertEqual(summary["arena_useful_bytes_peak"], 2500)
        self.assertEqual(summary["arena_stranded_bytes_peak"], 572)

    def test_combined_restore_is_not_reported_as_an_isolated_stage(self):
        summary = summarize(
            [
                {
                    "cache_hit": False,
                    "ttft_s": 0.2,
                    "restore_s": 0.03,
                    "store_s": 0.02,
                    "load_s": 0.0,
                    "transfer_s": 0.0,
                    "dequant_s": 0.0,
                }
            ]
        )

        self.assertEqual(summary["restore_mean_s"], 0.03)
        self.assertEqual(summary["store_mean_s"], 0.02)
        self.assertEqual(summary["load_mean_s"], 0.0)
        self.assertEqual(summary["transfer_mean_s"], 0.0)
        self.assertEqual(summary["dequant_mean_s"], 0.0)

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
                "reference_max_label_logit_delta": 0.03125,
                "reference_within_atol": True,
            },
            {
                "cache_hit": False,
                "article_bytes": 10,
                "ttft_s": 0.2,
                "predicted_label": "B",
                "gold_label": "A",
                "fp16_reference_label": "A",
                "reference_agreement": False,
                "reference_max_label_logit_delta": 0.125,
                "reference_within_atol": False,
            },
        ]
        summary = summarize(rows, cold_requests=1)
        self.assertEqual(summary["reference_label_agreement"], 0.5)
        self.assertEqual(summary["reference_label_mismatches"], 1)
        self.assertEqual(summary["reference_max_label_logit_delta"], 0.125)
        self.assertAlmostEqual(
            summary["reference_mean_max_label_logit_delta"], 0.078125
        )
        self.assertEqual(summary["reference_within_atol_rate"], 0.5)
        self.assertEqual(summary["reference_tolerance_violations"], 1)
        self.assertEqual(summary["fp16_reference_accuracy"], 1.0)
        self.assertEqual(summary["accuracy_delta_vs_fp16"], -0.5)
        self.assertEqual(summary["cold_hit_rate"], 1.0)
        self.assertEqual(summary["steady_state_hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
