from __future__ import annotations

import unittest

from src.quality_cache.data import QualityQuestion, QualityRequest
from src.quality_cache.inference.model import QualityModelRunner, ScoreResult
from src.quality_cache.reporting.metrics import summarize


class _CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode("utf-8"))


class SegmentedBaselineTest(unittest.TestCase):
    def test_segmented_baseline_reuses_only_l0_and_retains_no_article(self):
        runner = object.__new__(QualityModelRunner)
        runner.tokenizer = _CharacterTokenizer()
        runner.l0_cache = object()
        runner.prefill_cost_model = "test-measured"
        calls = {}

        def prefill(token_ids, past=None):
            calls["article_tokens"] = list(token_ids)
            calls["past"] = past
            return "transient-article-prefix"

        def score_suffix(suffix_ids, prefix, options=None):
            calls["suffix_tokens"] = list(suffix_ids)
            calls["prefix"] = prefix
            calls["options"] = options
            return ScoreResult("B", {"A": 0.0, "B": 1.0, "C": -1.0, "D": -2.0})

        runner._prefill = prefill
        runner.score_suffix = score_suffix
        runner._synchronize = lambda: None
        runner.estimate_article_bytes = lambda *args, **kwargs: 1234
        runner.memory_stats = lambda: {}

        question = QualityQuestion(
            question_id="q1",
            text="Which option?",
            options=("one", "two", "three", "four"),
            gold_label=1,
            difficult=False,
        )
        request = QualityRequest("q1", "a1", "Article text.", "hash", question)
        row = runner.serve_segmented_uncached(
            request,
            storage="accelerator-fp16",
            cache_strategy="document",
        )

        self.assertIs(calls["past"], runner.l0_cache)
        self.assertEqual(calls["prefix"], "transient-article-prefix")
        self.assertEqual(calls["options"], question.options)
        self.assertTrue(calls["article_tokens"])
        self.assertTrue(calls["suffix_tokens"])
        self.assertFalse(row["cache_hit"])
        self.assertTrue(row["root_only_hit"])
        self.assertEqual(row["matched_prefix_tokens"], 0)
        self.assertEqual(row["matched_prefill_tokens"], 0)
        self.assertEqual(row["avoided_prefill_tokens"], 0)
        self.assertEqual(row["cache_bytes"], 0)
        self.assertEqual(row["cached_documents"], 0)
        self.assertEqual(row["insertions"], 0)
        self.assertEqual(row["evictions"], 0)
        self.assertEqual(row["baseline_mode"], "segmented")
        self.assertEqual(row["inference_path"], "segmented-uncached")
        self.assertEqual(row["predicted_label"], "B")
        self.assertGreaterEqual(row["ttft_s"], row["prefill_s"])

        summary = summarize([row], cold_requests=1)
        self.assertEqual(summary["full_document_hit_rate"], 0.0)
        self.assertEqual(summary["article_token_hit_rate"], 0.0)
        self.assertEqual(summary["byte_hit_rate"], 0.0)
        self.assertEqual(summary["root_only_hit_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
