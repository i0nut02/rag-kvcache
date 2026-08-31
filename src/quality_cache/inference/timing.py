"""Typed per-request timing stages used by inference runners."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class StageTimings:
    """Disjoint cache-path stages plus the combined restore wall time.

    ``restore_s`` is intentionally the combined wall-clock diagnostic. The
    profiled restore path separately records source-to-device movement,
    dequantization/kernel execution, and final layer/block assembly in
    ``transfer_s``, ``dequant_s``, and ``load_s``. They are measured directly
    and therefore need not sum exactly to the outer Python wall time.
    ``store_s`` covers tensor conversion/storage and excludes cache-policy
    insertion, which remains in ``policy_s``.
    """

    lookup_s: float = 0.0
    load_s: float = 0.0
    transfer_s: float = 0.0
    dequant_s: float = 0.0
    restore_s: float = 0.0
    store_s: float = 0.0
    policy_s: float = 0.0
    prefill_s: float = 0.0

    def as_row(self) -> dict[str, float]:
        return asdict(self)
