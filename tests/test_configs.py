from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.quality_cache.matrix import build_matrix_commands, load_matrix


class ExperimentConfigTest(unittest.TestCase):
    def test_matrix_uses_requested_seed_budgets_and_storage(self):
        root = Path(__file__).resolve().parents[1]
        for name in (
            "cache_strategies.json",
            "confirmation.json",
            "lightweight.json",
            "standard.json",
        ):
            with self.subTest(name=name):
                payload = json.loads((root / "configs" / name).read_text(encoding="utf-8"))
                self.assertEqual(payload["seeds"], [42])
                self.assertEqual(payload["budget_mb"], [4096, 8192])
                self.assertEqual(
                    payload["storage"], ["accelerator-fp16", "cpu-int8"]
                )
                self.assertEqual(payload["split"], "test")
                self.assertEqual(
                    payload["dataset"],
                    "data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.test",
                )
                self.assertTrue(payload["no_inference"])
                self.assertFalse(payload["accuracy_available"])

    def test_fast_and_confirmation_profiles_use_random_limits(self):
        root = Path(__file__).resolve().parents[1]
        strategies = json.loads(
            (root / "configs" / "cache_strategies.json").read_text(encoding="utf-8")
        )
        self.assertEqual(strategies["policies"], ["lru", "lfu", "gdsf"])
        self.assertEqual(strategies["profiles"]["smoke"], {"limit": 10})
        self.assertEqual(strategies["profiles"]["confirmation"], {"limit": 50})
        self.assertEqual(strategies["profiles"]["full"], {"limit": None})
        lightweight = json.loads(
            (root / "configs" / "lightweight.json").read_text(encoding="utf-8")
        )
        confirmation = json.loads(
            (root / "configs" / "confirmation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(lightweight["workloads"], ["random"])
        self.assertEqual(lightweight["requests"], 10)
        self.assertEqual(confirmation["workloads"], ["random"])
        self.assertEqual(confirmation["requests"], 50)

    def test_primary_test_matrix_expands_every_combination(self):
        root = Path(__file__).resolve().parents[1]
        config = load_matrix(root / "configs" / "cache_strategies.json")
        commands = build_matrix_commands(config, "full", root / "results" / "matrix")
        self.assertEqual(config["primary_combinations"], 108)
        self.assertEqual(len(commands), 108)
        self.assertEqual(len({tuple(command) for command in commands}), 108)
        for command in commands:
            self.assertIn("--no-inference", command)
            self.assertNotIn("--limit", command)
            self.assertEqual(command[command.index("--split") + 1], "test")
            self.assertTrue(command[1].endswith(".htmlstripped.test"))

    def test_gpu_inference_keeps_cache_on_cuda(self):
        root = Path(__file__).resolve().parents[1]
        payload = json.loads(
            (root / "configs" / "gpu_inference.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["device"], "cuda")
        self.assertEqual(payload["storage"], "accelerator-fp16")
        self.assertEqual(payload["cache_device"], "cuda")
        self.assertFalse(payload["no_inference"])


if __name__ == "__main__":
    unittest.main()
