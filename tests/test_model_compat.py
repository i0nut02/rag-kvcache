from __future__ import annotations

import unittest

from src.quality_cache.inference.model import (
    default_reference_logit_atol,
    transformers_dtype_keyword,
)


class ModelCompatibilityTest(unittest.TestCase):
    def test_reference_tolerances_are_dtype_aware(self):
        self.assertEqual(default_reference_logit_atol("float16"), 0.0625)
        self.assertEqual(default_reference_logit_atol("bfloat16"), 0.25)
        self.assertEqual(default_reference_logit_atol("float32"), 0.001)

    def test_transformers_five_uses_new_dtype_keyword(self):
        self.assertEqual(transformers_dtype_keyword("4.48.3"), "torch_dtype")
        self.assertEqual(transformers_dtype_keyword("5.13.1"), "dtype")


if __name__ == "__main__":
    unittest.main()
