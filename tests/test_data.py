from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.quality_cache.data import (
    EXPECTED_COUNTS,
    grouped_max_hit_rate,
    load_quality_split,
    merge_quality_records,
)


def raw_question(index):
    return {
        "question_unique_id": f"q{index}",
        "question": f"Question {index}?",
        "options": ["a", "b", "c", "d"],
        "gold_label": index % 4 + 1,
        "difficult": index % 2 == 0,
    }


class QualityDataTest(unittest.TestCase):
    def test_two_writer_records_are_merged_without_question_loss(self):
        records = [
            {
                "article_id": "00100",
                "article": "A long article.",
                "set_unique_id": "left",
                "writer_id": "w1",
                "questions": [raw_question(1), raw_question(2)],
            },
            {
                "article_id": "00100",
                "article": "A long article.",
                "set_unique_id": "right",
                "writer_id": "w2",
                "questions": [raw_question(3)],
            },
        ]
        merged = merge_quality_records(records)
        self.assertEqual(len(merged), 1)
        self.assertEqual([item.question_id for item in merged[0].questions], ["q1", "q2", "q3"])

    def test_inconsistent_duplicate_article_is_rejected(self):
        records = [
            {"article_id": "1", "article": "first", "questions": []},
            {"article_id": "1", "article": "second", "questions": []},
        ]
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            merge_quality_records(records)

    def test_official_validation_requires_two_writer_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.train"
            path.write_text(json.dumps({
                "article_id": "1", "article": "only once", "questions": []
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "count mismatch|two writer records"):
                load_quality_split(path, split="train", verify_official_counts=True)

    def test_official_train_counts_and_total_constants(self):
        article_count, question_count = EXPECTED_COUNTS["train"]
        base, remainder = divmod(question_count, article_count)
        records = []
        question_index = 0
        for article_index in range(article_count):
            count = base + (article_index < remainder)
            left = count // 2
            questions = [raw_question(question_index + offset) for offset in range(count)]
            question_index += count
            for writer, subset in (("w1", questions[:left]), ("w2", questions[left:])):
                records.append({
                    "article_id": f"{article_index:05d}",
                    "article": f"article {article_index}",
                    "set_unique_id": f"{article_index}-{writer}",
                    "writer_id": writer,
                    "questions": subset,
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "QuALITY.v1.0.1.htmlstripped.train"
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            loaded = load_quality_split(path, split="train", verify_official_counts=True)
        self.assertEqual((len(loaded), sum(len(a.questions) for a in loaded)), (150, 2523))
        self.assertEqual(
            tuple(sum(values[index] for values in EXPECTED_COUNTS.values()) for index in (0, 1)),
            (381, 6737),
        )
        self.assertAlmostEqual((6737 - 381) / 6737, 0.9434466379694227)
        self.assertAlmostEqual(grouped_max_hit_rate(loaded), (2523 - 150) / 2523)


if __name__ == "__main__":
    unittest.main()
