from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.quality_cache.inference.reference import (
    attach_offline_reference,
    load_reference_jsonl,
)
from src.quality_cache.schema import RESULT_SCHEMA_VERSION


class OfflineReferenceTest(unittest.TestCase):
    def _reference(self):
        return {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "request_id": "q1",
            "article_id": "a1",
            "predicted_label": "B",
            "label_scores": {"A": 1.0, "B": 2.0, "C": 0.5, "D": -1.0},
            "gold_label": "B",
            "policy": "none",
            "storage": "accelerator-fp16",
            "execution_mode": "inference",
            "model": "model",
            "workload": "random",
            "seed": 42,
        }

    def test_reference_is_loaded_and_attached_without_a_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.jsonl"
            path.write_text(json.dumps(self._reference()) + "\n", encoding="utf-8")
            references = load_reference_jsonl(
                path,
                expected_model="model",
                expected_workload="random",
                expected_seed=42,
            )
        candidate = {
            "request_id": "q1",
            "article_id": "a1",
            "predicted_label": "B",
            "label_scores": {"A": 1.01, "B": 2.0, "C": 0.5, "D": -1.0},
            "gold_label": "B",
        }
        attach_offline_reference(
            candidate, references, storage="accelerator-fp16", agreement_atol=0.0625
        )
        self.assertTrue(candidate["reference_agreement"])
        self.assertAlmostEqual(candidate["reference_max_label_logit_delta"], 0.01)
        self.assertEqual(candidate["reference_mode"], "offline-jsonl")

    def test_fp16_delta_is_enforced_but_int8_delta_is_reported(self):
        reference = self._reference()
        references = {"q1": reference}
        candidate = {
            "request_id": "q1",
            "article_id": "a1",
            "predicted_label": "A",
            "label_scores": {"A": 3.0, "B": 2.0, "C": 0.5, "D": -1.0},
            "gold_label": "B",
        }
        with self.assertRaisesRegex(AssertionError, "offline-uncached mismatch"):
            attach_offline_reference(
                dict(candidate),
                references,
                storage="accelerator-fp16",
                agreement_atol=0.0625,
            )
        int8 = dict(candidate)
        attach_offline_reference(
            int8, references, storage="cpu-int8", agreement_atol=0.0625
        )
        self.assertFalse(int8["reference_agreement"])
        self.assertEqual(int8["reference_max_label_logit_delta"], 2.0)


if __name__ == "__main__":
    unittest.main()
