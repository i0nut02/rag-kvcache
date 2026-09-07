from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.quality_cache.data import QualityQuestion, QualityRequest
from src.quality_cache.inference.model import QualityModelRunner, ScoreResult
from src.quality_cache.inference.results import scored_request_fields, uncached_cache_fields
from src.quality_cache.inference.timing import StageTimings
from src.quality_cache.memory import memory_deltas, model_memory_snapshot
from src.quality_cache.schema import INFERENCE_TIMING_SCOPE, RESULT_SCHEMA_VERSION


class ModelReportingTest(unittest.TestCase):
    def test_scored_request_fields_preserve_labels_and_disjoint_timers(self):
        for gold_label in (1, None):
            with self.subTest(gold_label=gold_label):
                question = QualityQuestion("q", "Which?", ("a", "b", "c", "d"), gold_label, True)
                request = QualityRequest("q", "article", "text", "hash", question)
                score = ScoreResult("B", {"A": 0.0, "B": 1.0, "C": -1.0, "D": -2.0})
                timings = StageTimings(
                    lookup_s=0.1, load_s=0.2, transfer_s=0.3, dequant_s=0.4,
                    restore_s=0.95, store_s=0.5, policy_s=0.6, prefill_s=0.7,
                )
                self.assertEqual(scored_request_fields(request, score, timings, 3.0), {
                    "result_schema_version": RESULT_SCHEMA_VERSION,
                    "request_id": "q", "article_id": "article",
                    "lookup_s": 0.1, "load_s": 0.2, "transfer_s": 0.3,
                    "dequant_s": 0.4, "restore_s": 0.95, "store_s": 0.5,
                    "policy_s": 0.6, "prefill_s": 0.7, "ttft_s": 3.0,
                    "timing_scope": INFERENCE_TIMING_SCOPE,
                    "predicted_label": "B", "label_scores": score.scores,
                    "gold_label": "B" if gold_label is not None else None,
                    "difficult": True,
                })

    def test_baseline_cache_fields_retain_their_existing_schema(self):
        full = {
            "cache_bytes": 0, "metadata_bytes": 0, "cache_footprint_bytes": 0,
            "budget_bytes": 0, "occupancy": 0.0, "root_nodes": 0,
            "document_tree_nodes": 0, "cached_articles": 0, "insertions": 0,
            "evictions": 0, "policy": "none", "cache_strategy": "none",
        }
        self.assertEqual(uncached_cache_fields(segmented=False), full)
        self.assertEqual(uncached_cache_fields(segmented=True), {
            **full, "root_nodes": 1, "document_tree_nodes": 1,
            "useful_bytes": 0, "shared_bytes": 0, "stranded_bytes": 0,
            "cached_documents": 0, "cached_blocks": 0, "radix_nodes": 0,
            "cached_tokens": 0, "baseline_mode": "segmented",
            "inference_path": "segmented-uncached", "l0_reused": True,
        })

    def test_memory_snapshot_reads_only_the_selected_device(self):
        for device_type in ("cpu", "cuda", "mps"):
            with self.subTest(device=device_type):
                torch = SimpleNamespace(cuda=Mock(), mps=Mock())
                torch.cuda.memory_allocated.return_value = 20
                torch.cuda.memory_reserved.return_value = 30
                torch.mps.current_allocated_memory.return_value = 40
                torch.mps.driver_allocated_memory.return_value = 50
                device = SimpleNamespace(type=device_type)
                with patch("src.quality_cache.memory.current_process_rss_bytes", return_value=100):
                    snapshot = model_memory_snapshot(torch, device)
                self.assertEqual(snapshot, {
                    "process_rss_bytes": 100,
                    "cuda_allocated_bytes": 20 if device_type == "cuda" else 0,
                    "cuda_reserved_bytes": 30 if device_type == "cuda" else 0,
                    "mps_allocated_bytes": 40 if device_type == "mps" else 0,
                    "mps_driver_bytes": 50 if device_type == "mps" else 0,
                })
                if device_type == "cuda":
                    torch.cuda.memory_allocated.assert_called_once_with(device)
                    torch.cuda.memory_reserved.assert_called_once_with(device)
                else:
                    self.assertEqual(torch.cuda.mock_calls, [])
                if device_type == "mps":
                    torch.mps.current_allocated_memory.assert_called_once_with()
                    torch.mps.driver_allocated_memory.assert_called_once_with()
                else:
                    self.assertEqual(torch.mps.mock_calls, [])

    def test_memory_deltas_preserve_inputs_and_clamp_negative_differences(self):
        values = dict(zip(
            ("process_rss_bytes", "mps_allocated_bytes", "mps_driver_bytes",
             "cuda_allocated_bytes", "cuda_reserved_bytes"),
            (120, 0, 0, 80, 100),
        ))
        original = dict(values)
        baseline = dict.fromkeys(values, 100)
        original_baseline = dict(baseline)
        expected = {
            "process_rss_delta_bytes": 20, "mps_allocated_delta_bytes": 0,
            "mps_driver_delta_bytes": 0, "cuda_allocated_delta_bytes": 0,
            "cuda_reserved_delta_bytes": 0,
        }
        self.assertEqual(memory_deltas(values, baseline), expected)
        self.assertEqual(values, original)
        self.assertEqual(baseline, original_baseline)

        runner = QualityModelRunner.__new__(QualityModelRunner)
        runner._raw_memory_stats = lambda: dict(values)
        # Construction/testing paths without a baseline report zero deltas.
        self.assertEqual(runner.memory_stats(), {**values, **dict.fromkeys(expected, 0)})
        runner._memory_baseline = baseline
        self.assertEqual(runner.memory_stats(), {**values, **expected})


if __name__ == "__main__":
    unittest.main()
