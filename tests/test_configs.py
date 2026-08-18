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

    def test_selected_inference_confirmation_expands_expected_runs(self):
        root = Path(__file__).resolve().parents[1]
        config = load_matrix(root / "configs" / "inference_confirmation.json")

        smoke = build_matrix_commands(
            config, "smoke", root / "results" / "inference-smoke"
        )
        confirmation = build_matrix_commands(
            config, "confirmation", root / "results" / "inference-confirmation"
        )
        full = build_matrix_commands(
            config, "full", root / "results" / "inference-full"
        )

        self.assertEqual(len(smoke), 8)
        self.assertEqual(len({tuple(command) for command in smoke}), 8)
        self.assertTrue(all("--no-inference" not in command for command in smoke))
        self.assertTrue(all("--limit" in command for command in smoke))
        self.assertTrue(
            all(command[command.index("--limit") + 1] == "10" for command in smoke)
        )
        self.assertTrue(
            all(
                command[command.index("--limit") + 1] == "100"
                for command in confirmation
            )
        )
        self.assertTrue(all("--limit" not in command for command in full))
        self.assertTrue(
            all(command[command.index("--split") + 1] == "dev" for command in smoke)
        )

        uncached = [command for command in smoke if "none" in command]
        cached = [command for command in smoke if "none" not in command]
        self.assertEqual(len(uncached), 2)
        self.assertEqual(len(cached), 6)
        self.assertTrue(all("--budget-mb" not in command for command in uncached))
        self.assertTrue(all("--budget-mb" in command for command in cached))
        self.assertTrue(
            all("--reference-jsonl" not in command for command in uncached)
        )
        self.assertTrue(all("--reference-jsonl" in command for command in cached))
        self.assertTrue(
            all("--validate-agreement" not in command for command in smoke)
        )

    def test_segmented_baseline_reuses_existing_full_references(self):
        root = Path(__file__).resolve().parents[1]
        config = load_matrix(root / "configs" / "segmented_baseline_confirmation.json")
        output = root / "results" / "inference_confirmation" / "confirmation"
        commands = build_matrix_commands(config, "confirmation", output)

        self.assertEqual(len(commands), 4)
        segmented = [
            command
            for command in commands
            if command[command.index("--baseline-mode") + 1] == "segmented"
        ]
        self.assertEqual(len(segmented), 2)
        self.assertTrue(all("--policy" in command for command in segmented))
        self.assertTrue(
            all(command[command.index("--policy") + 1] == "none" for command in segmented)
        )
        self.assertTrue(all("--budget-mb" not in command for command in segmented))
        self.assertTrue(all("--reference-jsonl" in command for command in segmented))
        self.assertTrue(
            all(command[command.index("--limit") + 1] == "100" for command in commands)
        )

        references = {
            Path(command[command.index("--reference-jsonl") + 1]).name
            for command in segmented
        }
        self.assertEqual(
            references,
            {
                "dev_confirmation_01_uncached_random_fp16.jsonl",
                "dev_confirmation_06_uncached_zipf_fp16.jsonl",
            },
        )

    def test_fixed_block_sensitivity_is_a_six_run_test_only_sweep(self):
        root = Path(__file__).resolve().parents[1]
        config = load_matrix(root / "configs" / "fixed_block_sensitivity.json")
        output = root / "results" / "fixed-block-sensitivity"
        commands = build_matrix_commands(config, "full", output)

        self.assertEqual(len(commands), 6)
        self.assertTrue(all("--no-inference" in command for command in commands))
        self.assertTrue(all("--limit" not in command for command in commands))
        self.assertEqual(
            {
                int(command[command.index("--block-tokens") + 1])
                for command in commands
            },
            {16, 64, 256},
        )
        self.assertEqual(
            {command[command.index("--workload") + 1] for command in commands},
            {"random", "zipf"},
        )
        self.assertTrue(
            all(
                command[command.index("--budget-mb") + 1] == "4096"
                for command in commands
            )
        )

    def test_follow_up_inference_configs_keep_seed_42_and_selected_scope(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "int8_accuracy_confirmation.json": (2, "300"),
            "timing_repetitions.json": (8, "100"),
        }
        for name, (run_count, confirmation_limit) in expected.items():
            with self.subTest(name=name):
                config = load_matrix(root / "configs" / name)
                commands = build_matrix_commands(
                    config,
                    "confirmation",
                    root / "results" / name.removesuffix(".json"),
                )
                self.assertEqual(len(commands), run_count)
                self.assertTrue(
                    all("--no-inference" not in command for command in commands)
                )
                self.assertTrue(
                    all(
                        command[command.index("--seed") + 1] == "42"
                        for command in commands
                    )
                )
                self.assertTrue(
                    all(
                        command[command.index("--limit") + 1]
                        == confirmation_limit
                        for command in commands
                    )
                )


if __name__ == "__main__":
    unittest.main()
