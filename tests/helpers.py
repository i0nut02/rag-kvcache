from __future__ import annotations

from src.quality_cache.data import QualityArticle, QualityQuestion


def question(index: int, difficult: bool = False) -> QualityQuestion:
    return QualityQuestion(
        question_id=f"q{index}",
        text=f"Question {index}?",
        options=("one", "two", "three", "four"),
        gold_label=index % 4,
        difficult=difficult,
    )


def articles(count: int = 8, questions_each: int = 3) -> list[QualityArticle]:
    return [
        QualityArticle(
            article_id=f"a{article_index}",
            text=(f"Article {article_index} text. " * (article_index + 1)).strip(),
            questions=[
                question(article_index * questions_each + question_index)
                for question_index in range(questions_each)
            ],
        )
        for article_index in range(count)
    ]
