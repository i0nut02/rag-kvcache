from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


try:
    import torch
except ImportError:
    torch = None

from src.quality_cache.inference.tensors import (
    resolve_storage_device,
    restore_blocks,
    slice_stored_blocks,
    store_blocks,
    to_legacy,
    to_model_cache,
)


@unittest.skipIf(torch is None, "torch is not installed")
class TensorStorageTest(unittest.TestCase):
    def setUp(self):
        self.cache = tuple(
            (
                torch.randn(1, 2, 9, 8, dtype=torch.float16),
                torch.randn(1, 2, 9, 8, dtype=torch.float16),
            )
            for _ in range(3)
        )

    def test_fp16_blocks_round_trip_on_cpu(self):
        blocks = store_blocks(self.cache, "cpu-fp16", 4)
        restored = restore_blocks(blocks, dtype=torch.float16, device="cpu")
        self.assertEqual([block.token_count for block in blocks], [4, 4, 1])
        for original_layer, restored_layer in zip(self.cache, restored):
            for original, value in zip(original_layer, restored_layer):
                torch.testing.assert_close(original, value, rtol=0, atol=0)

    def test_int8_scales_bytes_and_expected_reduction(self):
        fp16 = store_blocks(self.cache, "cpu-fp16", 4)
        int8 = store_blocks(self.cache, "cpu-int8", 4)
        fp16_bytes = sum(block.stored_bytes for block in fp16)
        int8_bytes = sum(block.stored_bytes for block in int8)
        self.assertLess(int8_bytes, fp16_bytes * 0.60)
        first_quantized = int8[0].layers[0][0]
        self.assertEqual(tuple(first_quantized.scales.shape), (2,))
        restored = restore_blocks(int8, dtype=torch.float16, device="cpu")
        for original_layer, restored_layer in zip(self.cache, restored):
            for original, value in zip(original_layer, restored_layer):
                error = (original - value).abs().max().item()
                self.assertLess(error, 0.04)

    def test_stored_block_slice_spans_physical_boundaries(self):
        blocks = store_blocks(self.cache, "cpu-fp16", 4)
        sliced = slice_stored_blocks(blocks, 2, 8)
        restored = restore_blocks(sliced, dtype=torch.float16, device="cpu")
        self.assertEqual([block.token_count for block in sliced], [2, 4])
        for original_layer, restored_layer in zip(self.cache, restored):
            for original, value in zip(original_layer, restored_layer):
                torch.testing.assert_close(original[..., 2:8, :], value, rtol=0, atol=0)

    def test_accelerator_storage_requires_accelerator_device(self):
        with self.assertRaisesRegex(ValueError, "requires a CUDA or MPS device"):
            store_blocks(self.cache, "accelerator-fp16", 4)
        with self.assertRaisesRegex(ValueError, "requires a CUDA or MPS device"):
            resolve_storage_device("accelerator-fp16", "cpu")

    def test_transformers_v5_layer_cache_converts_to_legacy(self):
        layers = [
            SimpleNamespace(keys=key, values=value) for key, value in self.cache
        ]
        converted = to_legacy(SimpleNamespace(layers=layers))
        self.assertEqual(len(converted), len(self.cache))
        for original_layer, converted_layer in zip(self.cache, converted):
            for original, value in zip(original_layer, converted_layer):
                self.assertIs(original, value)

    def test_transformers_v5_legacy_cache_uses_constructor(self):
        class DynamicCacheV5:
            def __init__(self, data):
                self.data = data

        fake_transformers = SimpleNamespace(DynamicCache=DynamicCacheV5)
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            converted = to_model_cache(self.cache)
        self.assertIs(converted.data, self.cache)

    def test_transformers_v5_rejects_uninitialized_layer(self):
        cache = SimpleNamespace(layers=[SimpleNamespace(keys=None, values=None)])
        with self.assertRaisesRegex(TypeError, "not initialized"):
            to_legacy(cache)

    @unittest.skipUnless(torch is not None and torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_storage_smoke(self):
        source = tuple((key.to("mps"), value.to("mps")) for key, value in self.cache)
        blocks = store_blocks(
            source, "accelerator-fp16", 4, accelerator_device="mps"
        )
        restored = restore_blocks(blocks, dtype=torch.float16, device="mps")
        self.assertEqual(restored[0][0].device.type, "mps")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_storage_smoke(self):
        source = tuple((key.to("cuda"), value.to("cuda")) for key, value in self.cache)
        blocks = store_blocks(
            source, "accelerator-fp16", 4, accelerator_device="cuda"
        )
        restored = restore_blocks(blocks, dtype=torch.float16, device="cuda")
        self.assertEqual(restored[0][0].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
