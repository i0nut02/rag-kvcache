"""Result fields shared by the model runner's three execution paths.

These helpers only assemble completed measurements; they do not run inference,
read clocks, or change cache state. Keep their calls outside measured intervals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..data import QualityRequest
from ..schema import INFERENCE_TIMING_SCOPE, RESULT_SCHEMA_VERSION
from .timing import StageTimings

if TYPE_CHECKING:
    from .model import ScoreResult


def scored_request_fields(
    request: QualityRequest, score: ScoreResult, timings: StageTimings, ttft_s: float
) -> dict[str, Any]:
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "request_id": request.request_id,
        "article_id": request.article_id,
        **timings.as_row(),
        "ttft_s": ttft_s,
        "timing_scope": INFERENCE_TIMING_SCOPE,
        "predicted_label": score.label,
        "label_scores": score.scores,
        "gold_label": request.question.answer_letter,
        "difficult": request.question.difficult,
    }


def uncached_cache_fields(*, segmented: bool) -> dict[str, Any]:
    """Preserve the full and segmented baselines' existing result fields.

    The segmented baseline pins L0 but retains no article KV. The full-forward
    baseline has no cache tree and historically emits fewer cache diagnostics.
    """
    fields = {
        "cache_bytes": 0,
        "metadata_bytes": 0,
        "cache_footprint_bytes": 0,
        "budget_bytes": 0,
        "occupancy": 0.0,
        "root_nodes": int(segmented),
        "document_tree_nodes": int(segmented),
        "cached_articles": 0,
        "insertions": 0,
        "evictions": 0,
        "policy": "none",
        "cache_strategy": "none",
    }
    if segmented:
        fields.update(
            useful_bytes=0,
            shared_bytes=0,
            stranded_bytes=0,
            cached_documents=0,
            cached_blocks=0,
            radix_nodes=0,
            cached_tokens=0,
            baseline_mode="segmented",
            inference_path="segmented-uncached",
            l0_reused=True,
        )
    return fields
