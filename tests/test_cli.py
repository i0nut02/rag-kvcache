import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.quality_cache.cli import main
from src.quality_cache.schema import RESULT_SCHEMA_VERSION


class CLITest(unittest.TestCase):
    def test_simulation_writes_summary_and_manifest(self):
        records = []
        for article_index in range(4):
            for writer_index in range(2):
                records.append({
                    "article_id": str(article_index),
                    "article": f"article text {article_index}",
                    "set_unique_id": f"{article_index}-{writer_index}",
                    "questions": [{
                        "question_unique_id": f"q-{article_index}-{writer_index}",
                        "question": "Which option?",
                        "options": ["a", "b", "c", "d"],
                        "gold_label": 1,
                    }],
                })
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "tiny.train"
            output = Path(directory) / "summary.csv"
            dataset.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            result = main([
                "simulate", str(dataset), "--split", "train",
                "--workloads", "grouped", "--policies", "lru", "farthest-next-use",
                "--seeds", "17", "--budget-percent", "100",
                "--tokenizer-mode", "approximate",
                "--output", str(output),
            ])
            self.assertEqual(result, 0)
            self.assertTrue(output.with_suffix(".csv.manifest.json").exists())
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["policy"] for row in rows}, {"lru", "farthest-next-use"}
            )
            self.assertEqual(
                {row["result_schema_version"] for row in rows},
                {RESULT_SCHEMA_VERSION},
            )

    def test_collect_rejects_legacy_results(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.csv"
            output = Path(directory) / "combined.csv"
            legacy.write_text("policy,request_hit_rate\nlru,0.5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "legacy or mixed"):
                main(["collect", str(legacy), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
