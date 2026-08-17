from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .data import QualityArticle, QualityQuestion


PROMPT_VERSION = "quality-mc-v1"
SYSTEM_PROMPT = (
    "Read the article and answer the multiple-choice question. "
    "Respond with exactly one label: A, B, C, or D."
)


def l0_text() -> str:
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\nArticle:\n"
    )


def article_tail(article_text: str) -> str:
    return f"{article_text}\n\n"


def suffix_text(question: QualityQuestion) -> str:
    choices = "\n".join(
        f"{label}. {option}" for label, option in zip("ABCD", question.options)
    )
    return (
        f"Question: {question.text}\n{choices}\n"
        "Answer:<|im_end|>\n<|im_start|>assistant\n"
    )


def full_prompt(article: QualityArticle, question: QualityQuestion) -> str:
    return l0_text() + article_tail(article.text) + suffix_text(question)


def encode_parts(tokenizer, article_text: str, question: QualityQuestion):
    """Tokenize exact concatenated parts and assert the cache boundary is stable."""
    l0_ids = tokenizer.encode(l0_text(), add_special_tokens=False)
    article_ids = tokenizer.encode(article_tail(article_text), add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix_text(question), add_special_tokens=False)
    whole = tokenizer.encode(
        l0_text() + article_tail(article_text) + suffix_text(question),
        add_special_tokens=False,
    )
    combined = list(l0_ids) + list(article_ids) + list(suffix_ids)
    if combined != list(whole):
        raise ValueError(
            "tokenizer merges across a prompt boundary; use prefix lengths from full-prompt encoding"
        )
    return list(l0_ids), list(article_ids), list(suffix_ids)


@dataclass(frozen=True)
class PromptIdentity:
    article_id: str
    article_hash: str
    model_revision: str
    tokenizer_revision: str
    dtype: str
    quantization: str
    prompt_version: str = PROMPT_VERSION

    def digest(self) -> str:
        raw = "\x1f".join(str(value) for value in self.__dict__.values())
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
