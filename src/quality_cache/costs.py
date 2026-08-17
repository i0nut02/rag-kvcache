from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .schema import PREFILL_CALIBRATION_SCHEMA_VERSION


@dataclass(frozen=True)
class PrefillCostModel:
    """Monotone piecewise-linear model fitted to measured article prefills."""

    samples: tuple[tuple[int, float], ...]
    source: str

    @classmethod
    def linear(cls, tokens_per_second: float = 50_000.0) -> "PrefillCostModel":
        if tokens_per_second <= 0:
            raise ValueError("tokens_per_second must be positive")
        return cls(
            ((1, 1.0 / tokens_per_second),),
            f"linear-token-estimate-{tokens_per_second:g}-tokens-per-second",
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "PrefillCostModel":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("calibration_schema_version")
        if version != PREFILL_CALIBRATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported prefill calibration schema {version!r}; "
                f"expected {PREFILL_CALIBRATION_SCHEMA_VERSION!r}"
            )
        raw_samples = payload.get("samples", [])
        if not raw_samples:
            raise ValueError("prefill calibration must contain at least one sample")
        by_tokens: dict[int, list[float]] = {}
        for sample in raw_samples:
            tokens = int(sample["tokens"])
            seconds = float(sample["seconds"])
            if tokens <= 0 or seconds <= 0:
                raise ValueError("prefill calibration samples must be positive")
            by_tokens.setdefault(tokens, []).append(seconds)
        ordered = []
        previous = 0.0
        for tokens in sorted(by_tokens):
            # Duplicate token lengths are averaged and measurement noise is made
            # monotone so that a larger prefill can never have a lower cost.
            seconds = max(previous, sum(by_tokens[tokens]) / len(by_tokens[tokens]))
            ordered.append((tokens, seconds))
            previous = seconds
        return cls(tuple(ordered), f"measured-piecewise:{path.resolve()}")

    def predict(self, tokens: int) -> float:
        tokens = int(tokens)
        if tokens <= 0:
            return 0.0
        points = ((0, 0.0),) + self.samples
        for (left_tokens, left_s), (right_tokens, right_s) in zip(points, points[1:]):
            if tokens <= right_tokens:
                span = max(1, right_tokens - left_tokens)
                fraction = (tokens - left_tokens) / span
                return left_s + fraction * (right_s - left_s)
        if len(points) >= 3:
            left_tokens, left_s = points[-2]
            right_tokens, right_s = points[-1]
            slope = (right_s - left_s) / max(1, right_tokens - left_tokens)
        else:
            right_tokens, right_s = points[-1]
            slope = right_s / max(1, right_tokens)
        return right_s + max(0.0, slope) * (tokens - right_tokens)

    def incremental(self, total_tokens: int, cached_tokens: int = 0) -> float:
        if cached_tokens < 0 or cached_tokens > total_tokens:
            raise ValueError("cached_tokens must be within the total token count")
        return max(0.0, self.predict(total_tokens) - self.predict(cached_tokens))


def load_prefill_cost_model(path: str | Path | None) -> PrefillCostModel:
    return PrefillCostModel.from_file(path) if path else PrefillCostModel.linear()
