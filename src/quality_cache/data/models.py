"""Typed QuALITY articles, questions, and serving requests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class QualityQuestion:
    question_id: str
    text: str
    options: tuple[str, str, str, str]
    gold_label: int | None
    difficult: bool
    writer_id: str = ""

    @property
    def answer_letter(self) -> str | None:
        return None if self.gold_label is None else "ABCD"[self.gold_label]


@dataclass
class QualityArticle:
    article_id: str
    text: str
    title: str = ""
    questions: list[QualityQuestion] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QualityRequest:
    request_id: str
    article_id: str
    article_text: str
    article_hash: str
    question: QualityQuestion
