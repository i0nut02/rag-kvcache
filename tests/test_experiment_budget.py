from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.quality_cache.experiment_budget import BudgetExpired, ExperimentBudget


class ExperimentBudgetTest(unittest.TestCase):
    def test_deadline_survives_new_instance_and_cell_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            budget = ExperimentBudget(path, limit_s=10)
            self.assertEqual(budget.remaining(), 10)
            self.assertFalse(path.exists())  # Reading doesn't start the clock.
            with patch("src.quality_cache.experiment_budget.time.time", return_value=100):
                self.assertEqual(budget._start(), 10)
            with patch("src.quality_cache.experiment_budget.time.time", return_value=104):
                self.assertEqual(ExperimentBudget(path, limit_s=10)._start(), 6)
            with patch("src.quality_cache.experiment_budget.time.time", return_value=110):
                with patch("src.quality_cache.experiment_budget.subprocess.Popen") as popen:
                    with self.assertRaises(BudgetExpired):
                        budget.run(["must-not-launch"], log_path=Path(directory) / "log")
                    popen.assert_not_called()
            with self.assertRaises(ValueError):
                ExperimentBudget(path, limit_s=9).remaining()
        for limit in (-1, 0, 36001, float("nan")):
            with self.assertRaises(ValueError):
                ExperimentBudget(Path("unused"), limit_s=limit)

    @unittest.skipUnless(os.name == "posix", "Colab deadline uses POSIX process groups")
    def test_silent_stalled_child_is_killed_at_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = ExperimentBudget(Path(directory) / "budget.json", limit_s=0.2)
            start = time.monotonic()
            with self.assertRaises(BudgetExpired):
                budget.run([sys.executable, "-c", "import time; time.sleep(10)"],
                           log_path=Path(directory) / "log")
            self.assertLess(time.monotonic() - start, 3)
            self.assertEqual(budget.remaining(), 0)

    @unittest.skipUnless(os.name == "posix", "Colab deadline uses POSIX process groups")
    def test_successful_process_logs_and_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "log"
            budget = ExperimentBudget(Path(directory) / "budget.json", limit_s=10)
            elapsed = budget.run([sys.executable, "-c", "print('finished')"], log_path=log)
            self.assertGreater(elapsed, 0)
            self.assertIn("finished", log.read_text())
            with self.assertRaises(subprocess.CalledProcessError):
                budget.run([sys.executable, "-c", "raise SystemExit(7)"], log_path=log)
