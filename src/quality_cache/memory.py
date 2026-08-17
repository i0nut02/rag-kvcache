from __future__ import annotations

import resource
import sys


_PSUTIL_PROCESS = None


def current_process_rss_bytes() -> int:
    """Return current RSS when psutil is present, otherwise peak RSS."""
    try:
        import psutil

        global _PSUTIL_PROCESS
        if _PSUTIL_PROCESS is None:
            _PSUTIL_PROCESS = psutil.Process()
        return int(_PSUTIL_PROCESS.memory_info().rss)
    except ImportError:
        # macOS reports bytes; Linux reports KiB.
        factor = 1 if sys.platform == "darwin" else 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * factor)


def nonnegative_delta(current: int, baseline: int) -> int:
    return max(0, int(current) - int(baseline))
