from __future__ import annotations

import json
import unittest
from pathlib import Path


class SlurmJobTest(unittest.TestCase):
    def test_cuda_job_matches_requested_experiment(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "slurm" / "quality_cuda_100.sbatch").read_text(
            encoding="utf-8"
        )
        required = (
            "#SBATCH --gres=gpu:1",
            "#SBATCH --mem=32G",
            "--device cuda",
            "--storage accelerator-fp16",
            "--budget-mb 4096",
            "--workload random",
            "--seed 42",
            "--limit 100",
        )
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, script)
        self.assertNotIn("--no-inference", script)

    def test_gpu_config_and_slurm_job_agree(self):
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
