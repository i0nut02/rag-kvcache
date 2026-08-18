from __future__ import annotations

import unittest


try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
except (ImportError, RuntimeError):
    torch = None
    GPT2Config = GPT2LMHeadModel = None

from src.quality_cache.caches import CacheKey, StoredKV, new_prefix_cache
from src.quality_cache.data import QualityQuestion, QualityRequest
from src.quality_cache.inference.model import QualityModelRunner
from src.quality_cache.inference.tensors import (
    restore_blocks,
    store_blocks,
    to_legacy,
    to_model_cache,
)
from src.quality_cache.prompt import l0_text


class _ModuloTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [byte % 64 for byte in text.encode("utf-8")]


@unittest.skipIf(torch is None or GPT2LMHeadModel is None, "torch/transformers unavailable")
class CPUIntegrationTest(unittest.TestCase):
    def test_segmented_runner_matches_full_prompt_without_retaining_article(self):
        torch.manual_seed(5)
        runner = object.__new__(QualityModelRunner)
        runner.torch = torch
        runner.device = torch.device("cpu")
        runner.dtype = torch.float32
        runner.tokenizer = _ModuloTokenizer()
        runner.model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=64,
                bos_token_id=None,
                eos_token_id=None,
                n_positions=512,
                n_embd=32,
                n_layer=2,
                n_head=4,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        ).eval()
        runner.model_revision = "tiny-gpt2"
        runner.tokenizer_revision = "modulo-tokenizer"
        runner.prefill_cost_model = "measured-online"
        runner.l0_ids = runner.tokenizer.encode(l0_text(), add_special_tokens=False)
        runner.l0_cache = runner._prefill(runner.l0_ids)
        runner.l0_tokens = len(runner.l0_ids)
        runner._memory_baseline = runner._raw_memory_stats()

        question = QualityQuestion(
            question_id="q1",
            text="Which option?",
            options=("one", "two", "three", "four"),
            gold_label=0,
            difficult=False,
        )
        request = QualityRequest("q1", "a1", "Short article.", "hash", question)
        full = runner.score_uncached(request)
        segmented = runner.serve_segmented_uncached(
            request,
            storage="cpu-fp16",
            cache_strategy="document",
        )

        self.assertEqual(segmented["predicted_label"], full.label)
        for label in "ABCD":
            self.assertAlmostEqual(
                segmented["label_scores"][label], full.scores[label], places=5
            )
        self.assertEqual(segmented["matched_prefix_tokens"], 0)
        self.assertEqual(segmented["cache_bytes"], 0)
        self.assertEqual(segmented["cached_documents"], 0)

    def test_uncached_and_fp16_cached_logits_and_labels_agree(self):
        torch.manual_seed(7)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=64,
                bos_token_id=None,
                eos_token_id=None,
                n_positions=64,
                n_embd=32,
                n_layer=2,
                n_head=4,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        ).eval()
        prefix_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        suffix_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)
        full_ids = torch.cat((prefix_ids, suffix_ids), dim=1)
        with torch.inference_mode():
            full_logits = model(full_ids, use_cache=False).logits[:, -1]
            prefix = to_legacy(model(prefix_ids, use_cache=True).past_key_values)
        blocks = store_blocks(prefix, "cpu-fp16", block_tokens=4)
        restored = restore_blocks(blocks, dtype=torch.float32, device="cpu")
        attention_mask = torch.ones((1, full_ids.shape[1]), dtype=torch.long)
        with torch.inference_mode():
            cached_logits = model(
                suffix_ids,
                attention_mask=attention_mask,
                past_key_values=to_model_cache(restored),
                use_cache=False,
            ).logits[:, -1]
        torch.testing.assert_close(full_logits, cached_logits, rtol=2e-3, atol=2e-3)
        self.assertEqual(full_logits.argmax(-1).item(), cached_logits.argmax(-1).item())

    def test_radix_partial_prefix_logits_match_uncached(self):
        torch.manual_seed(11)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=64,
                bos_token_id=None,
                eos_token_id=None,
                n_positions=64,
                n_embd=32,
                n_layer=2,
                n_head=4,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        ).eval()
        key = CacheKey("a", "hash-a", "model", "tokenizer", "prompt", "float32", "none")
        source_ids = [1, 2, 3, 4]
        target_ids = [1, 2, 5, 6]
        suffix_ids = [7, 8]
        with torch.inference_mode():
            source_cache = to_legacy(
                model(torch.tensor([source_ids]), use_cache=True).past_key_values
            )
        stored = store_blocks(source_cache, "cpu-fp16", block_tokens=4)
        cache = new_prefix_cache("radix", 1_000_000, l0=object())
        cache.insert(key, source_ids, StoredKV(4, blocks=stored), 1.0)
        match = cache.lookup(
            CacheKey("b", "hash-b", "model", "tokenizer", "prompt", "float32", "none"),
            target_ids,
        )
        self.assertEqual(match.matched_tokens, 2)
        restored = restore_blocks(match.blocks, dtype=torch.float32, device="cpu")
        remaining = torch.tensor([target_ids[match.matched_tokens:]], dtype=torch.long)
        with torch.inference_mode():
            target_cache = to_legacy(
                model(
                    remaining,
                    attention_mask=torch.ones((1, len(target_ids)), dtype=torch.long),
                    past_key_values=to_model_cache(restored),
                    use_cache=True,
                ).past_key_values
            )
            cached_logits = model(
                torch.tensor([suffix_ids]),
                attention_mask=torch.ones((1, len(target_ids) + len(suffix_ids)), dtype=torch.long),
                past_key_values=to_model_cache(target_cache),
                use_cache=False,
            ).logits[:, -1]
            full_logits = model(
                torch.tensor([target_ids + suffix_ids]), use_cache=False
            ).logits[:, -1]
        torch.testing.assert_close(full_logits, cached_logits, rtol=2e-3, atol=2e-3)
        self.assertEqual(full_logits.argmax(-1).item(), cached_logits.argmax(-1).item())


if __name__ == "__main__":
    unittest.main()
