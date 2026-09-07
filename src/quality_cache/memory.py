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


def model_memory_snapshot(torch, device) -> dict[str, int]:
    """Sample RSS and the selected device without importing torch here.

    Accelerator values are current allocator samples, not intra-forward peaks.
    Unsupported/inactive devices retain the existing zero-valued fields.
    """
    values = {
        "process_rss_bytes": current_process_rss_bytes(),
        "mps_allocated_bytes": 0,
        "mps_driver_bytes": 0,
        "cuda_allocated_bytes": 0,
        "cuda_reserved_bytes": 0,
    }
    if device.type == "mps":
        values["mps_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        values["mps_driver_bytes"] = int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        values["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
        values["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
    return values


def memory_deltas(values: dict[str, int], baseline: dict[str, int]) -> dict[str, int]:
    """Return nonnegative deltas for the five model-memory snapshot fields."""
    return {
        f"{name}_delta_bytes": nonnegative_delta(
            values[f"{name}_bytes"], baseline[f"{name}_bytes"]
        )
        for name in (
            "process_rss", "mps_allocated", "mps_driver", "cuda_allocated", "cuda_reserved"
        )
    }
