from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from .models import QualityArticle, QualityRequest
from .quality import flatten_requests


DEFAULT_SEEDS = (17, 23, 42)


def build_workload(
    articles: Iterable[QualityArticle],
    kind: str,
    *,
    seed: int = 17,
    requests: int | None = None,
    zipf_exponent: float = 1.1,
) -> list[QualityRequest]:
    articles = list(articles)
    natural = flatten_requests(articles)
    count = requests if requests is not None else len(natural)
    rng = random.Random(seed)
    if kind == "grouped":
        return _cycle(natural, count)
    if kind == "random":
        shuffled = list(natural)
        rng.shuffle(shuffled)
        return _cycle(shuffled, count)
    if kind == "zipf":
        return _sample_articles(articles, count, rng, zipf_exponent)
    raise ValueError(f"unsupported workload: {kind}")


def _cycle(values: list[QualityRequest], count: int) -> list[QualityRequest]:
    if not values and count:
        raise ValueError("cannot construct a workload from no questions")
    return [values[index % len(values)] for index in range(count)] if values else []


def _sample_articles(
    articles: list[QualityArticle],
    count: int,
    rng: random.Random,
    exponent: float,
) -> list[QualityRequest]:
    usable = [article for article in articles if article.questions]
    weights = [1.0 / ((rank + 1) ** exponent) for rank in range(len(usable))]
    cursors: defaultdict[str, int] = defaultdict(int)
    result = []
    for _ in range(count):
        article = rng.choices(usable, weights=weights, k=1)[0]
        question = article.questions[cursors[article.article_id] % len(article.questions)]
        cursors[article.article_id] += 1
        result.append(_request(article, question))
    return result
def _request(article, question) -> QualityRequest:
    return QualityRequest(
        request_id=question.question_id,
        article_id=article.article_id,
        article_text=article.text,
        article_hash=article.content_hash,
        question=question,
    )
