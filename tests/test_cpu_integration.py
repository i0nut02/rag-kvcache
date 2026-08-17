from __future__ import annotations

import unittest


try:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
except (ImportError, RuntimeError):
    torch = None
    GPT2Config = GPT2LMHeadModel = None

from src.quality_cache.caches import CacheKey, StoredKV, new_prefix_cache
from src.quality_cache.inference.tensors import (
    restore_blocks,
    store_blocks,
    to_legacy,
    to_model_cache,
)


@unittest.skipIf(torch is None or GPT2LMHeadModel is None, "torch/transformers unavailable")
class CPUIntegrationTest(unittest.TestCase):
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
