"""Reuse an uncached inference trace as an offline numerical reference."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..schema import RESULT_SCHEMA_VERSION


LABELS = "ABCD"


def load_reference_jsonl(
    path: str | Path,
    *,
    expected_model: str | None = None,
    expected_workload: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Load and validate an uncached FP16 per-request result file."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"reference JSONL does not exist: {path}")
    references: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid reference JSONL at {path}:{line_number}"
                ) from error
            _validate_reference_row(
                row,
                path=path,
                line_number=line_number,
                expected_model=expected_model,
                expected_workload=expected_workload,
                expected_seed=expected_seed,
            )
            request_id = str(row["request_id"])
            if request_id in references:
                raise ValueError(f"duplicate reference request_id: {request_id}")
            references[request_id] = row
    if not references:
        raise ValueError(f"reference JSONL is empty: {path}")
    return references


def attach_offline_reference(
    row: dict[str, Any],
    references: dict[str, dict[str, Any]],
    *,
    storage: str,
    agreement_atol: float,
) -> None:
    """Attach agreement fields without executing a second model forward."""
    request_id = str(row.get("request_id"))
    reference = references.get(request_id)
    if reference is None:
        raise ValueError(f"request {request_id!r} is absent from the reference JSONL")
    if row.get("article_id") != reference.get("article_id"):
        raise ValueError(f"article mismatch for reference request {request_id!r}")
    if row.get("gold_label") != reference.get("gold_label"):
        raise ValueError(f"gold-label mismatch for reference request {request_id!r}")

    candidate_scores = _label_scores(row, f"candidate request {request_id!r}")
    reference_scores = _label_scores(reference, f"reference request {request_id!r}")
    candidate_label = row.get("predicted_label")
    reference_label = reference.get("predicted_label")
    agreement = candidate_label == reference_label
    logit_delta = max(
        abs(candidate_scores[label] - reference_scores[label]) for label in LABELS
    )
    row.update(
        {
            "uncached_label": reference_label,
            "fp16_reference_label": reference_label,
            "reference_agreement": agreement,
            "reference_max_label_logit_delta": logit_delta,
            "reference_logit_atol": agreement_atol,
            "reference_mode": "offline-jsonl",
        }
    )
    if storage != "cpu-int8" and (not agreement or logit_delta > agreement_atol):
        raise AssertionError(
            "cached/offline-uncached mismatch: "
            f"request={request_id}, label agreement={agreement}, "
            f"max label-logit delta={logit_delta:.6g}, atol={agreement_atol:.6g}"
        )


def _validate_reference_row(
    row: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    expected_model: str | None,
    expected_workload: str | None,
    expected_seed: int | None,
) -> None:
    where = f"{path}:{line_number}"
    if row.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"reference schema mismatch at {where}")
    if row.get("execution_mode") != "inference" or row.get("policy") != "none":
        raise ValueError(f"reference row must be uncached inference at {where}")
    if row.get("storage") != "accelerator-fp16":
        raise ValueError(f"reference row must use accelerator-fp16 at {where}")
    if not row.get("request_id"):
        raise ValueError(f"reference row has no request_id at {where}")
    if row.get("predicted_label") not in LABELS:
        raise ValueError(f"reference row has an invalid predicted label at {where}")
    _label_scores(row, f"reference row at {where}")
    expected = {
        "model": expected_model,
        "workload": expected_workload,
        "seed": expected_seed,
    }
    for field, value in expected.items():
        if value is not None and row.get(field) != value:
            raise ValueError(
                f"reference {field} mismatch at {where}: "
                f"expected {value!r}, got {row.get(field)!r}"
            )


def _label_scores(row: dict[str, Any], description: str) -> dict[str, float]:
    scores = row.get("label_scores")
    if not isinstance(scores, dict):
        raise ValueError(f"{description} has no label_scores object")
    parsed = {}
    for label in LABELS:
        try:
            value = float(scores[label])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{description} has no numeric {label} score") from error
        if not math.isfinite(value):
            raise ValueError(f"{description} has a non-finite {label} score")
        parsed[label] = value
    return parsed
