from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.quality_cache.reporting.inference_analysis import (
    RUN_SPECS,
    analyze_inference_confirmation,
    bootstrap_ratio_ci,
)


class InferenceAnalysisTest(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic(self):
        first = bootstrap_ratio_ci([2.0, 2.0], [1.0, 1.0], samples=100, seed=7)
        second = bootstrap_ratio_ci([2.0, 2.0], [1.0, 1.0], samples=100, seed=7)

        self.assertEqual(first, (2.0, 2.0))
        self.assertEqual(first, second)

    def test_analysis_validates_suite_and_writes_fair_comparisons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            output = root / "analysis"
            results.mkdir()
            for spec in RUN_SPECS:
                path = results / f"dev_confirmation_{spec.name}.jsonl"
                rows = [self._row(spec, index) for index in range(2)]
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                path.with_suffix(".jsonl.manifest.json").write_text(
                    json.dumps(
                        {
                            "dataset_checksum": "test-checksum",
                            "git_revision": "test-revision",
                            "hardware": {"torch": "test-torch"},
                        }
                    ),
                    encoding="utf-8",
                )

            with patch(
                "src.quality_cache.reporting.inference_analysis._make_figures",
                return_value=[],
            ):
                artifacts = analyze_inference_confirmation(
                    results,
                    output,
                    bootstrap_samples=100,
                    seed=42,
                )

            self.assertEqual(len(artifacts), 7)
            with (output / "fair_speedups.csv").open(encoding="utf-8") as handle:
                comparisons = list(csv.DictReader(handle))
            self.assertEqual(len(comparisons), 6)
            self.assertTrue(
                all(float(row["cache_only_speedup"]) == 2.0 for row in comparisons)
            )
            self.assertTrue(
                all(float(row["end_to_end_speedup"]) == 6.0 for row in comparisons)
            )

            diagnostics = json.loads(
                (output / "analysis.json").read_text(encoding="utf-8")
            )
            self.assertTrue(diagnostics["validation"]["traces_aligned"])
            self.assertTrue(diagnostics["validation"]["segmented_invariants"])
            self.assertEqual(
                diagnostics["validation"]["requests_by_workload"],
                {"random": 2, "zipf": 2},
            )
            self.assertEqual(len(diagnostics["mismatch_details"]), 2)

    @staticmethod
    def _row(spec, index: int) -> dict:
        segmented_label = "B" if spec.workload == "zipf" and index == 1 else "A"
        if spec.kind == "full":
            predicted_label = "A"
        elif spec.storage == "cpu-int8" and index == 0:
            predicted_label = "B"
        else:
            predicted_label = segmented_label
        label_scores = (
            {"A": 0.1, "B": 0.3, "C": 0.0, "D": -0.1}
            if predicted_label == "B"
            else {"A": 0.3, "B": 0.1, "C": 0.0, "D": -0.1}
        )
        cache_hit = spec.kind == "cache" and index == 1
        matched_tokens = 10 if cache_hit else 0
        cache_bytes = 1_000 if spec.kind == "cache" else 0
        row = {
            "request_index": index,
            "request_id": f"{spec.workload}-request-{index}",
            "article_id": f"{spec.workload}-article-{index}",
            "workload": spec.workload,
            "model": "test-model",
            "cache_strategy": spec.strategy,
            "policy": spec.policy,
            "storage": spec.storage,
            "cache_hit": cache_hit,
            "cache_hit_ratio": 0.5 if cache_hit else 0.0,
            "matched_prefix_tokens": matched_tokens,
            "matched_prefill_tokens": matched_tokens,
            "avoided_prefill_tokens": matched_tokens,
            "matched_cache_bytes": 100 if cache_hit else 0,
            "cache_bytes": cache_bytes,
            "cached_documents": 1 if spec.kind == "cache" else 0,
            "insertions": 1 if spec.kind == "cache" else 0,
            "evictions": 0,
            "article_tokens": 10,
            "article_bytes": 100,
            "total_prompt_tokens": 12,
            "cached_prompt_tokens": matched_tokens,
            "ttft_s": {"full": 6.0, "segmented": 2.0, "cache": 1.0}[spec.kind],
            "prefill_s": 0.0,
            "predicted_label": predicted_label,
            "gold_label": "A",
            "label_scores": label_scores,
            "difficult": True,
            "cold_requests": 1,
        }
        if spec.kind == "segmented":
            row.update(
                {
                    "baseline_mode": "segmented",
                    "inference_path": "segmented-uncached",
                    "root_only_hit": True,
                }
            )
        return row


if __name__ == "__main__":
    unittest.main()
