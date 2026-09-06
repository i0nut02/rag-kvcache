"""A persistent wall-clock deadline for the bounded Colab experiment.

The window starts with the first benchmark process. Pauses and resumed runtimes
share the same deadline; re-running a cell cannot grant another ten hours.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Sequence


class BudgetExpired(RuntimeError):
    """The experiment window is over; completed results remain valid."""


class ExperimentBudget:
    def __init__(self, path: Path, *, limit_s: float = 10 * 3600):
        if not 0 < limit_s <= 10 * 3600:
            raise ValueError("experiment limit must be positive and at most ten hours")
        self.path = path
        self.limit_s = limit_s

    def remaining(self) -> float:
        if not self.path.exists():
            return self.limit_s
        record = json.loads(self.path.read_text())
        if record["limit_s"] != self.limit_s:
            raise ValueError("cannot change the saved experiment time budget")
        return max(0.0, record["deadline_unix_s"] - time.time())

    def _start(self) -> float:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x") as handle:
                now = time.time()
                json.dump({"started_unix_s": now, "deadline_unix_s": now + self.limit_s,
                           "limit_s": self.limit_s}, handle, indent=2)
        except FileExistsError:
            pass
        remaining = self.remaining()
        if remaining <= 0:
            raise BudgetExpired("Ten-hour experiment deadline reached; no new process started.")
        return remaining

    def run(self, command: Sequence[str], *, log_path: Path, cwd: Path | None = None) -> float:
        """Stream output, killing the entire process group when time expires.

        The watchdog runs even if inference stops printing progress. Timeout and
        keyboard interruption kill the child before returning to the notebook.
        POSIX process groups are supported on the target Colab/Linux runtime.
        """
        if os.name != "posix":
            raise RuntimeError("bounded experiment execution requires POSIX process groups")
        remaining = self._start()
        start = time.monotonic()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        expired = threading.Event()
        with log_path.open("a") as log:
            process = subprocess.Popen(
                list(command), cwd=cwd, start_new_session=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )

            def kill_group():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            def expire():
                expired.set()
                kill_group()

            watchdog = threading.Timer(max(0, remaining - (time.monotonic() - start)), expire)
            watchdog.daemon = True
            watchdog.start()
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
                if expired.is_set():
                    raise BudgetExpired("Experiment deadline reached; interrupted run is NOT complete.")
                if status:
                    raise subprocess.CalledProcessError(status, list(command))
            finally:
                watchdog.cancel()
                watchdog.join()
                if process.poll() is None:
                    kill_group()
                process.wait()
                process.stdout.close()
        return time.monotonic() - start
