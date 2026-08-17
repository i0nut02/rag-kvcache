"""QuALITY records, loading, deduplication, and validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .models import QualityArticle, QualityQuestion, QualityRequest


EXPECTED_COUNTS = {
    "train": (150, 2523),
    "dev": (115, 2086),
    "test": (116, 2128),
}
def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _parse_label(value: Any) -> int | None:
    if value in (None, "", -1):
        return None
    label = int(value)
    if not 1 <= label <= 4:
        raise ValueError(f"gold_label must be 1..4, got {value!r}")
    return label - 1


def merge_quality_records(records: Iterable[dict[str, Any]]) -> list[QualityArticle]:
    """Merge the two writer records per article without dropping questions."""
    merged: dict[str, QualityArticle] = {}
    for record in records:
        article_id = str(record["article_id"])
        text = str(record["article"])
        article = merged.get(article_id)
        if article is None:
            article = QualityArticle(
                article_id=article_id,
                text=text,
                title=str(record.get("title", "")),
            )
            merged[article_id] = article
        elif article.text != text:
            raise ValueError(f"article {article_id} has inconsistent text across writer records")

        writer_id = str(record.get("writer_id", ""))
        set_id = str(record.get("set_unique_id", writer_id or article_id))
        for index, raw in enumerate(record.get("questions", [])):
            options = tuple(str(option) for option in raw["options"])
            if len(options) != 4:
                raise ValueError(f"question {set_id}:{index} does not have four options")
            question_id = str(
                raw.get("question_unique_id")
                or raw.get("question_id")
                or f"{set_id}:{index}"
            )
            article.questions.append(
                QualityQuestion(
                    question_id=question_id,
                    text=str(raw["question"]),
                    options=options,  # type: ignore[arg-type]
                    gold_label=_parse_label(raw.get("gold_label")),
                    difficult=_as_bool(raw.get("difficult", False)),
                    writer_id=writer_id,
                )
            )
    return list(merged.values())


def load_quality_split(
    path: str | Path,
    *,
    split: str | None = None,
    verify_official_counts: bool = False,
) -> list[QualityArticle]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    articles = merge_quality_records(records)
    if verify_official_counts:
        inferred = split or infer_split(path)
        expected = EXPECTED_COUNTS.get(inferred)
        if expected is None:
            raise ValueError(f"cannot verify counts for unrecognized split {inferred!r}")
        actual = (len(articles), sum(len(article.questions) for article in articles))
        if actual != expected:
            raise ValueError(f"{inferred} count mismatch: expected {expected}, found {actual}")
        record_counts = Counter(str(record["article_id"]) for record in records)
        invalid = [article_id for article_id, count in record_counts.items() if count != 2]
        if invalid:
            raise ValueError(
                f"official QuALITY splits require two writer records per article; "
                f"found invalid counts for {len(invalid)} article(s)"
            )
    return articles


def infer_split(path: Path) -> str:
    name = path.name.lower()
    for split in EXPECTED_COUNTS:
        if name.endswith(f".{split}") or f".{split}." in name:
            return split
    return ""


def flatten_requests(articles: Iterable[QualityArticle]) -> list[QualityRequest]:
    requests: list[QualityRequest] = []
    for article in articles:
        for question in article.questions:
            requests.append(
                QualityRequest(
                    request_id=question.question_id,
                    article_id=article.article_id,
                    article_text=article.text,
                    article_hash=article.content_hash,
                    question=question,
                )
            )
    return requests


def dataset_checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grouped_max_hit_rate(articles: Iterable[QualityArticle]) -> float:
    counts = [len(article.questions) for article in articles]
    return sum(max(0, count - 1) for count in counts) / max(1, sum(counts))
