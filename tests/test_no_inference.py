from __future__ import annotations

import contextlib
import io
import types
import time
import unittest
from unittest.mock import patch

from src.quality_cache.cli import build_parser
from src.quality_cache.inference.no_inference import NoInferenceRunner
from src.quality_cache.schema import RESULT_SCHEMA_VERSION
from src.quality_cache.data import build_workload
from tests.helpers import articles


class _CharacterTokenizer:
    init_kwargs = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]


class NoInferenceTest(unittest.TestCase):
    def test_flag_and_strategy_are_exposed_by_run_parser(self):
        args = build_parser().parse_args(
            [
                "run",
                "tiny.train",
                "--split",
                "train",
                "--no-inference",
                "--cache-strategy",
                "radix",
                "--agreement-atol",
                "0.1",
                "--budget-mb",
                "1",
                "--output",
                "out.jsonl",
            ]
        )
        self.assertTrue(args.no_inference)
        self.assertEqual(args.cache_strategy, "radix")
        self.assertEqual(args.storage, "accelerator-fp16")
        self.assertEqual(args.agreement_atol, 0.1)
        self.assertIsNone(args.requests)

    def test_limit_alias_sets_positive_request_count(self):
        args = build_parser().parse_args(
            [
                "run",
                "tiny.train",
                "--split",
                "train",
                "--limit",
                "10",
                "--budget-mb",
                "1",
                "--output",
                "out.jsonl",
            ]
        )
        self.assertEqual(args.requests, 10)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(
                    [
                        "run",
                        "tiny.train",
                        "--split",
                        "train",
                        "--limit",
                        "0",
                        "--budget-mb",
                        "1",
                        "--output",
                        "out.jsonl",
                    ]
                )

    def test_runner_never_loads_model_weights_or_calls_forward(self):
        config = types.SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=1,
            num_attention_heads=2,
            hidden_size=16,
            _commit_hash="config-revision",
        )
        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=_CharacterTokenizer()),
            patch("transformers.AutoConfig.from_pretrained", return_value=config),
            patch("transformers.AutoModelForCausalLM.from_pretrained") as load_model,
        ):
            runner = NoInferenceRunner("fake-model", dtype="float16")
            trace = build_workload(articles(1, 2), "grouped")
            cache = runner.new_cache(1_000_000, "lru", strategy="document")
            first = runner.serve(trace[0], cache, storage="cpu-fp16")
            second = runner.serve(trace[1], cache, storage="cpu-fp16")
        load_model.assert_not_called()
        self.assertFalse(runner.model_weights_loaded)
        self.assertEqual(first["result_schema_version"], RESULT_SCHEMA_VERSION)
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertGreater(first["article_cache_hit_ratio"], 0.0)
        self.assertLess(first["article_cache_hit_ratio"], 1.0)
        self.assertEqual(first["article_text_cache_hit_ratio"], 0.0)
        self.assertEqual(second["article_cache_hit_ratio"], 1.0)
        self.assertEqual(second["article_text_cache_hit_ratio"], 1.0)
        self.assertTrue(first["partial_document_tree_hit"])
        self.assertTrue(first["partial_article_hit"])
        self.assertFalse(first["partial_article_text_hit"])
        self.assertTrue(first["root_only_hit"])
        self.assertFalse(second["root_only_hit"])
        self.assertGreater(first["document_tree_hit_ratio"], 0.0)
        self.assertLess(first["document_tree_hit_ratio"], 1.0)
        self.assertEqual(second["document_tree_hit_ratio"], 1.0)
        self.assertEqual(
            first["document_tree_hit_ratio"],
            first["document_tree_cached_tokens"] / first["document_tree_total_tokens"],
        )
        self.assertGreater(first["cache_hit_ratio"], 0.0)
        self.assertLess(first["cache_hit_ratio"], second["cache_hit_ratio"])
        self.assertLess(second["cache_hit_ratio"], 1.0)
        self.assertEqual(
            second["cache_hit_ratio"],
            second["cached_prompt_tokens"] / second["total_prompt_tokens"],
        )
        self.assertIsNone(second["predicted_label"])
        self.assertLessEqual(second["cache_bytes"], second["budget_bytes"])
        self.assertGreater(second["metadata_bytes"], 0)
        self.assertGreaterEqual(
            second["cache_footprint_bytes"], second["cache_bytes"]
        )
        self.assertGreater(second["process_rss_bytes"], 0)

    def test_twenty_thousand_requests_complete_under_one_minute(self):
        runner = NoInferenceRunner.__new__(NoInferenceRunner)
        runner.model_name = "fake-model"
        runner.dtype_name = "float16"
        runner.tokenizer = _CharacterTokenizer()
        runner.model_revision = "model"
        runner.tokenizer_revision = "tokenizer"
        runner.layers = 2
        runner.kv_heads = 1
        runner.head_dim = 8
        runner._article_token_cache = {}
        trace = build_workload(articles(40, 4), "zipf", requests=20_000, seed=42)
        started = time.perf_counter()
        for strategy in ("document", "fixed-block", "radix"):
            cache = runner.new_cache(
                200_000, "lru", strategy=strategy, block_tokens=16
            )
            for request in trace:
                row = runner.serve(
                    request,
                    cache,
                    storage="cpu-fp16",
                    block_tokens=16,
                    cache_strategy=strategy,
                )
                self.assertLessEqual(row["cache_bytes"], row["budget_bytes"])
        self.assertLess(time.perf_counter() - started, 60.0)


if __name__ == "__main__":
    unittest.main()
