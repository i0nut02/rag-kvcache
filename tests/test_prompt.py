import re
import unittest

from src.quality_cache.prompt import PromptIdentity, encode_parts
from tests.helpers import articles


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return re.findall(r"<\|[^|]+\|>|\w+|[^\w\s]", text)


class PromptTest(unittest.TestCase):
    def test_stable_prefix_parts_equal_full_prompt_tokenization(self):
        article = articles(1, 1)[0]
        parts = encode_parts(Tokenizer(), article.text, article.questions[0])
        self.assertTrue(all(parts))

    def test_identity_digest_changes_with_content_or_quantization(self):
        base = dict(
            article_id="a", article_hash="h", model_revision="m",
            tokenizer_revision="t", dtype="float16", quantization="none",
        )
        first = PromptIdentity(**base).digest()
        second = PromptIdentity(**{**base, "article_hash": "changed"}).digest()
        third = PromptIdentity(**{**base, "quantization": "int8"}).digest()
        self.assertEqual(len({first, second, third}), 3)


if __name__ == "__main__":
    unittest.main()
