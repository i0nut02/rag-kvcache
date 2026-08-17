from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.quality_cache.costs import PrefillCostModel
from src.quality_cache.schema import PREFILL_CALIBRATION_SCHEMA_VERSION


class PrefillCostModelTest(unittest.TestCase):
    def test_piecewise_interpolation_and_incremental_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "calibration_schema_version": PREFILL_CALIBRATION_SCHEMA_VERSION,
                        "samples": [
                            {"tokens": 100, "seconds": 1.0},
                            {"tokens": 200, "seconds": 3.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            model = PrefillCostModel.from_file(path)
        self.assertAlmostEqual(model.predict(50), 0.5)
        self.assertAlmostEqual(model.predict(150), 2.0)
        self.assertAlmostEqual(model.predict(250), 4.0)
        self.assertAlmostEqual(model.incremental(200, 100), 2.0)

    def test_noisy_samples_are_made_monotone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "calibration_schema_version": PREFILL_CALIBRATION_SCHEMA_VERSION,
                        "samples": [
                            {"tokens": 100, "seconds": 2.0},
                            {"tokens": 200, "seconds": 1.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            model = PrefillCostModel.from_file(path)
        self.assertEqual(model.predict(200), 2.0)


if __name__ == "__main__":
    unittest.main()
