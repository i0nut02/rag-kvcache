from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from src.quality_cache.matrix import build_matrix_commands, _validate_selected_runs
from src.quality_cache.reporting.metrics import summarize
from src.quality_cache.reporting.repetitions import (
    analyze_repetitions, artifact_paths, read_repetition_run, sha256,
)
from src.quality_cache.schema import RESULT_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]


class StrategyRepetitionsTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/strategy_repetitions.json").read_text())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_protocol_scope_budget_balanced_order_and_references(self):
        config = self.config
        _validate_selected_runs(config)
        self.assertEqual(config["benchmark_limit_hours"], 10)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["profiles"], {"smoke": {"limit": 10}, "confirmation": {"limit": 1000}})
        commands = build_matrix_commands(config, "confirmation", self.directory)
        self.assertEqual(len(commands), 20)
        self.assertEqual(len({tuple(c) for c in commands}), 20)
        self.assertEqual(sum(r["policy"] == "none" for r in config["runs"]), 2)
        for spec, command in zip(config["runs"], commands):
            self.assertNotIn("--no-inference", command)
            self.assertEqual(command[command.index("--limit") + 1], "1000")
            self.assertEqual(command[command.index("--model-revision") + 1], config["model_revision"])
            if spec["policy"] == "none":
                self.assertEqual(command[command.index("--baseline-mode") + 1], "segmented")
                self.assertNotIn("--budget-mb", command)
            else:
                self.assertEqual(command[command.index("--budget-mb") + 1], "4096")
                self.assertEqual(spec["policy"], "lru")
                self.assertEqual(command[command.index("--block-tokens") + 1], "256")
                self.assertEqual(spec["reference_run"], f"reference_{spec['workload']}_segmented")
        for workload in ("random", "zipf"):
            orders = [[r["cache_strategy"] for r in config["runs"]
                       if r["workload"] == workload and r["repetition"] == repetition]
                      for repetition in (1, 2, 3)]
            self.assertEqual(orders, [
                ["document", "fixed-block", "radix"],
                ["fixed-block", "radix", "document"],
                ["radix", "document", "fixed-block"],
            ])
        with self.assertRaisesRegex(ValueError, "not configured"):
            build_matrix_commands(config, "full", self.directory)

    def test_notebook_is_clean_python_with_deadline_and_no_full_profile(self):
        notebook = json.loads((ROOT / "notebooks/strategy_repetitions_colab.ipynb").read_text())
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), f"cell-{index}", "exec")
        source = "\n".join("".join(c["source"]) for c in notebook["cells"])
        self.assertIn("budget.run(", source)
        self.assertIn("simulation-first", source)
        self.assertIn('PROFILE = "confirmation"', source)
        self.assertNotIn('PROFILE = "full"', source)
        self.assertNotIn("subprocess.Popen(", source)  # Must go through the tested deadline helper.
        self.assertIn("ALLOW_INCOMPLETE", source)

    def _path(self, spec):
        return self.directory / f"dev_confirmation_{spec['name']}.jsonl"

    def _fixture(self):
        config = self.config
        config["profiles"]["confirmation"]["limit"] = 4
        for spec in config["runs"]:
            cached = spec["policy"] != "none"
            strategy = spec["cache_strategy"] if cached else "none"
            fields = {
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "execution_mode": "inference", "model": config["model"],
                "seed": 42, "workload": spec["workload"], "policy": spec["policy"],
                "storage": "accelerator-fp16", "device": "cuda", "cache_strategy": strategy,
                "budget_bytes": config["budget_mb"] * 2**20 if cached else 0, "block_tokens": 256,
            }
            scale = {"none": 2, "document": 1, "fixed-block": 1.25, "radix": 1.5}[strategy]
            rows = [{
                **fields, "request_index": index,
                # A repeated Zipf Q&A is valid at a different position.
                "request_id": str(index % 2) if spec["workload"] == "zipf" else str(index),
                "article_id": "article", "article_tokens": 10, "total_prompt_tokens": 15,
                "matched_prefix_tokens": 10 if cached else 0,
                "cache_bytes": 100 if cached else 0, "cache_hit": cached,
                "gold_label": "A", "predicted_label": "A", "difficult": index % 2 == 0,
                "ttft_s": scale * (index + 1), "baseline_mode": None if cached else "segmented",
                "inference_path": "article-cache" if cached else "segmented-uncached",
            } for index in range(4)]
            path = self._path(spec)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            path.with_suffix(".summary.json").write_text(json.dumps({
                **summarize(rows), **fields, "split": "dev",
            }))
            manifest = {
                **fields, "split": "dev", "cache_strategy": spec["cache_strategy"],
                "storage_format": "accelerator-fp16", "dtype": "float16", "kv_backend": "tensor",
                "model_revision": config["model_revision"], "tokenizer_revision": config["model_revision"],
                "dataset_checksum": "checksum", "prompt_version": "prompt", "git_revision": "commit",
                "timing_scope": "inference", "hardware": {
                    "python": "3.12", "torch": "test", "torch_cuda": "12", "cuda_device_name": "Tesla T4",
                    "cuda_device_total_memory": 15 * 2**30, "cuda_device_capability": [7, 5],
                },
            }
            if cached:
                ref = self.directory / f"dev_confirmation_{spec['reference_run']}.jsonl"
                manifest["reference_checksum"] = sha256(ref)
            path.with_suffix(".jsonl.manifest.json").write_text(json.dumps(manifest))

    def _analyze(self, **kwargs):
        return analyze_repetitions(self.config, self.directory, self.directory / "analysis", **kwargs)

    def test_analysis_pairs_run_means_not_requests_and_accepts_zipf_duplicates(self):
        self._fixture()
        result = self._analyze()
        self.assertEqual(len(result["run_metrics"]), 40)  # 20 runs, two scopes.
        self.assertEqual(len(result["paired_runs"]), 36)  # 6 groups, three cache pairs, two scopes.
        doc = next(r for r in result["strategy_variation"] if r["workload"] == "random"
                   and r["strategy"] == "document" and r["scope"] == "all_requests")
        self.assertEqual(doc["repetitions"], 3)
        self.assertEqual(doc["ttft_mean_s_median"], 2.5)
        self.assertAlmostEqual(doc["ttft_p90_s_median"], 3.7)
        self.assertEqual(doc["ttft_mean_s_stdev"], 0)
        pair = next(r for r in result["paired_variation"] if r["baseline"] == "document"
                    and r["candidate"] == "fixed-block")
        self.assertEqual(pair["mean_reduction_percent_median"], -25)
        self.assertEqual(pair["candidate_faster_runs"], 0)
        self.assertTrue(all(r["baseline"] != "segmented" for r in result["paired_runs"]))
        self.assertTrue((self.directory / "analysis/run_metrics.csv").is_file())

    def test_repeated_baseline_is_not_counted_as_three_measurements(self):
        self._fixture()
        references = [r for r in self._analyze()["strategy_variation"] if r["strategy"] == "segmented"]
        self.assertTrue(all(r["repetitions"] == 1 for r in references))
        self.assertTrue(all(math.isnan(r["ttft_mean_s_stdev"]) for r in references))

    def test_variability_uses_run_level_percentiles_not_pooled_requests(self):
        self._fixture()
        spec = next(r for r in self.config["runs"] if r["name"] == "rep3_random_document")
        path = self._path(spec)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["ttft_s"] *= 2
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        results = self._analyze()["strategy_variation"]
        measured = next(r for r in results if r["workload"] == "random"
                        and r["strategy"] == "document" and r["scope"] == "all_requests")
        self.assertEqual(measured["repetitions"], 3)
        self.assertEqual(measured["ttft_mean_s_median"], 2.5)
        self.assertEqual(measured["ttft_mean_s_max"], 5)
        self.assertAlmostEqual(measured["ttft_mean_s_stdev"], math.sqrt(25 / 12))
        self.assertAlmostEqual(measured["ttft_p90_s_median"], 3.7)
        self.assertAlmostEqual(measured["ttft_p90_s_max"], 7.4)

    def test_partial_analysis_drops_whole_incomplete_group(self):
        self._fixture()
        spec = self.config["runs"][-1]
        for path in artifact_paths(self._path(spec)):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "incomplete run"):
            self._analyze()
        result = self._analyze(allow_incomplete=True)
        self.assertFalse(any(r["workload"] == "zipf" and r["repetition"] == 3
                             for r in result["run_metrics"]))
        self.assertTrue(all(r["repetitions"] == 2 for r in result["paired_variation"] if r["workload"] == "zipf"))
        provenance = json.loads((self.directory / "analysis/analysis.json").read_text())
        self.assertEqual(provenance["missing_runs"], [spec["name"]])
        self.assertEqual(len(provenance["excluded_unpaired_runs"]), 2)

    def test_rejects_corrupt_provenance_and_rows(self):
        self._fixture()
        spec = self.config["runs"][-1]
        path = self._path(spec)
        manifest_path = path.with_suffix(".jsonl.manifest.json")
        original_manifest = json.loads(manifest_path.read_text())
        for key, value in (
            ("git_revision", "different"), ("dataset_checksum", "different"),
            ("model_revision", "different"), ("result_schema_version", "quality-kv-v3"),
            ("reference_checksum", "wrong"),
        ):
            with self.subTest(key=key):
                manifest_path.write_text(json.dumps({**original_manifest, key: value}))
                with self.assertRaises(ValueError):
                    self._analyze()
        wrong_gpu = copy.deepcopy(original_manifest)
        wrong_gpu["hardware"]["cuda_device_name"] = "A100"
        manifest_path.write_text(json.dumps(wrong_gpu))
        with self.assertRaisesRegex(ValueError, "runtime manifests"):
            self._analyze()
        manifest_path.write_text(json.dumps(original_manifest))
        original_rows = [json.loads(line) for line in path.read_text().splitlines()]
        for key, value in (("request_index", 5), ("article_id", "wrong"), ("ttft_s", float("nan")),
                           ("cache_bytes", 10**15), ("result_schema_version", "quality-kv-v3")):
            with self.subTest(key=key):
                rows = copy.deepcopy(original_rows)
                rows[0][key] = value
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                with self.assertRaises(ValueError):
                    self._analyze()
        path.write_text(json.dumps(original_rows[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "expected 4 requests"):
            read_repetition_run(path, spec, self.config, "confirmation")
