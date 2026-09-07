from __future__ import annotations

import json
import unittest
from pathlib import Path


class GPUConfigTest(unittest.TestCase):
    def test_gpu_config_matches_requested_experiment(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "gpu_inference.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["device"], "cuda")
        self.assertEqual(config["cache_device"], "cuda")
        self.assertEqual(config["storage"], "accelerator-fp16")
        self.assertEqual(config["budget_mb"], 4096)
        self.assertEqual(config["workload"], "random")
        self.assertEqual(config["limit"], 100)
        self.assertFalse(config["no_inference"])


if __name__ == "__main__":
    unittest.main()
