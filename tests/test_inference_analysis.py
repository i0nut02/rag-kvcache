from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.quality_cache.reporting.inference_analysis import (
    ConfirmationRun,
    RUN_SPECS,
    _comparison_plot_labels,
    _validate_suite,
    analyze_inference_confirmation,
    bootstrap_ratio_ci,
    load_analysis_suite,
)
from src.quality_cache.schema import RESULT_SCHEMA_VERSION


class InferenceAnalysisTest(unittest.TestCase):
    def test_speedup_plot_labels_distinguish_physical_backends(self):
        runs = (
            ConfirmationRun(
                "tensor",
                "Document tensor FP16",
                "Tensor FP16",
                "random",
                "cache",
                "document",
                "lru",
                "accelerator-fp16",
            ),
            ConfirmationRun(
                "triton",
                "Document CPU INT8, Triton restore",
                "INT8 Triton",
                "random",
                "cache",
                "document",
                "lru",
                "cpu-int8",
            ),
        )
        labels = _comparison_plot_labels(
            [
                {"run": "tensor", "workload": "random"},
                {"run": "triton", "workload": "random"},
            ],
            runs,
        )
        self.assertEqual(labels, ["random: Tensor FP16", "random: INT8 Triton"])

    def test_arena_triton_analysis_suite_covers_six_aligned_paths(self):
        root = Path(__file__).resolve().parents[1]
        runs = load_analysis_suite(root / "configs" / "arena_triton_analysis.json")
        self.assertEqual(len(runs), 6)
        self.assertEqual(sum(run.kind == "segmented" for run in runs), 1)
        self.assertEqual(sum(run.storage == "cpu-int8" for run in runs), 2)
        self.assertEqual(len({run.workload for run in runs}), 1)

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
                            "result_schema_version": RESULT_SCHEMA_VERSION,
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
            report = (output / "results.md").read_text(encoding="utf-8")
            self.assertIn("| Workload | Path | Storage |", report)
            self.assertIn("Document LRU FP16", report)

    def test_custom_suite_does_not_require_a_full_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "suite.json"
            path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "name": "segmented",
                                "label": "Segmented",
                                "short_label": "Segmented",
                                "workload": "random",
                                "kind": "segmented",
                                "strategy": "none",
                                "policy": "none",
                                "storage": "accelerator-fp16",
                            },
                            {
                                "name": "document",
                                "label": "Document",
                                "short_label": "Document",
                                "workload": "random",
                                "kind": "cache",
                                "strategy": "document",
                                "policy": "lru",
                                "storage": "accelerator-fp16",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            runs = load_analysis_suite(path)
            results = root / "results"
            output = root / "analysis"
            results.mkdir()
            for spec in runs:
                result = results / f"dev_confirmation_{spec.name}.jsonl"
                result.write_text(
                    "".join(
                        json.dumps(self._row(spec, index)) + "\n"
                        for index in range(2)
                    ),
                    encoding="utf-8",
                )
                result.with_suffix(".jsonl.manifest.json").write_text(
                    json.dumps(
                        {
                            "dataset_checksum": "test-checksum",
                            "result_schema_version": RESULT_SCHEMA_VERSION,
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
                    suite_config=path,
                )
            self.assertEqual([run.name for run in runs], ["segmented", "document"])
            self.assertEqual(len(artifacts), 6)
            with (output / "fair_speedups.csv").open(encoding="utf-8") as handle:
                comparison = next(csv.DictReader(handle))
            self.assertEqual(float(comparison["cache_only_speedup"]), 2.0)
            self.assertEqual(comparison["end_to_end_speedup"], "nan")

    def test_analysis_rejects_mixed_result_schemas(self):
        segmented = ConfirmationRun(
            "segmented",
            "Segmented",
            "Segmented",
            "random",
            "segmented",
            "none",
            "none",
            "accelerator-fp16",
        )
        document = ConfirmationRun(
            "document",
            "Document",
            "Document",
            "random",
            "cache",
            "document",
            "lru",
            "accelerator-fp16",
        )
        rows = {
            segmented.name: [self._row(segmented, 0)],
            document.name: [
                {
                    **self._row(document, 0),
                    "result_schema_version": "quality-kv-v2",
                }
            ],
        }
        manifests = {
            name: {
                "dataset_checksum": "test-checksum",
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "hardware": {"torch": "test-torch"},
            }
            for name in rows
        }

        with self.assertRaisesRegex(ValueError, "mixed result schemas"):
            _validate_suite(rows, manifests, (segmented, document))

    def test_analysis_rejects_manifest_schema_mismatch(self):
        segmented = ConfirmationRun(
            "segmented",
            "Segmented",
            "Segmented",
            "random",
            "segmented",
            "none",
            "none",
            "accelerator-fp16",
        )
        rows = {segmented.name: [self._row(segmented, 0)]}
        manifests = {
            segmented.name: {
                "dataset_checksum": "test-checksum",
                "result_schema_version": "quality-kv-v2",
                "hardware": {"torch": "test-torch"},
            }
        }

        with self.assertRaisesRegex(ValueError, "incompatible result schemas"):
            _validate_suite(rows, manifests, (segmented,))

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
            "result_schema_version": RESULT_SCHEMA_VERSION,
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
