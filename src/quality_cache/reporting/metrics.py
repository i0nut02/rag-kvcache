from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable

from ..schema import RESULT_SCHEMA_VERSION


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - location) + ordered[upper] * (location - lower)


def summarize(rows: Iterable[dict[str, Any]], *, cold_requests: int = 0) -> dict[str, Any]:
    rows = list(rows)
    hits = [bool(row["cache_hit"]) for row in rows]
    hit_ratios = [
        float(row.get("cache_hit_ratio", float(row["cache_hit"]))) for row in rows
    ]
    partial_hits = [bool(row.get("partial_cache_hit", False)) for row in rows]
    partial_article_hits = [
        bool(row.get("partial_article_hit", False)) for row in rows
    ]
    partial_article_text_hits = [
        bool(row.get("partial_article_text_hit", row.get("partial_article_hit", False)))
        for row in rows
    ]
    article_text_hit_ratios = [
        float(row.get("article_text_cache_hit_ratio", row.get("article_cache_hit_ratio", row["cache_hit"])))
        for row in rows
    ]
    document_tree_hit_ratios = [
        float(row.get("document_tree_hit_ratio", row.get("article_cache_hit_ratio", row["cache_hit"])))
        for row in rows
    ]
    partial_document_tree_hits = [
        bool(row.get("partial_document_tree_hit", False)) for row in rows
    ]
    root_only_hits = [bool(row.get("root_only_hit", False)) for row in rows]
    document_tree_tokens = sum(
        int(row.get("document_tree_total_tokens", row.get("article_tokens", 0)))
        for row in rows
    )
    document_tree_cached_tokens = sum(
        int(row.get("document_tree_cached_tokens", row.get("matched_prefix_tokens", 0)))
        for row in rows
    )
    requested_tokens = sum(
        int(row.get("total_prompt_tokens", row.get("article_tokens", 0)))
        for row in rows
    )
    matched_tokens = sum(
        int(row.get("cached_prompt_tokens", row.get("matched_prefix_tokens", 0)))
        for row in rows
    )
    requested_bytes = sum(int(row.get("article_bytes", 0)) for row in rows)
    requested_article_tokens = sum(int(row.get("article_tokens", 0)) for row in rows)
    matched_article_tokens = sum(
        int(row.get("matched_prefix_tokens", 0)) for row in rows
    )
    hit_bytes = sum(
        int(
            row.get(
                "matched_cache_bytes",
                int(row.get("article_bytes", 0)) if row["cache_hit"] else 0,
            )
        )
        for row in rows
    )
    ttft = [float(row["ttft_s"]) for row in rows if float(row.get("ttft_s", -1)) >= 0]
    labels = [row for row in rows if row.get("gold_label") is not None and row.get("predicted_label")]
    hard = [row for row in labels if row.get("difficult")]
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "requests": len(rows),
        "distinct_articles": len(
            {row.get("article_id") for row in rows if row.get("article_id") is not None}
        ),
        # Macro average: each request contributes a value in [0, 1].
        "request_hit_rate": statistics.fmean(hit_ratios) if hit_ratios else math.nan,
        "cache_hit_ratio_mean": statistics.fmean(hit_ratios) if hit_ratios else math.nan,
        "full_document_hit_rate": sum(hits) / max(1, len(hits)),
        "partial_prefix_hit_rate": sum(partial_hits) / max(1, len(partial_hits)),
        "partial_article_hit_rate": (
            sum(partial_article_hits) / max(1, len(partial_article_hits))
        ),
        "partial_article_text_hit_rate": (
            sum(partial_article_text_hits) / max(1, len(partial_article_text_hits))
        ),
        "article_text_hit_rate": (
            statistics.fmean(article_text_hit_ratios)
            if article_text_hit_ratios else math.nan
        ),
        "document_tree_hit_rate": (
            statistics.fmean(document_tree_hit_ratios)
            if document_tree_hit_ratios else math.nan
        ),
        "document_tree_token_hit_rate": (
            document_tree_cached_tokens / max(1, document_tree_tokens)
        ),
        "partial_document_tree_hit_rate": (
            sum(partial_document_tree_hits) / max(1, len(partial_document_tree_hits))
        ),
        "root_only_hit_rate": sum(root_only_hits) / max(1, len(root_only_hits)),
        # Micro average over article tokens only. This is the direct measure of
        # reusable document-prefill work and excludes L0 and the query suffix.
        "article_token_hit_rate": (
            matched_article_tokens / max(1, requested_article_tokens)
        ),
        "requested_article_tokens": requested_article_tokens,
        # Micro average: token-weighted across the complete prompt workload.
        "prefix_token_hit_rate": matched_tokens / max(1, requested_tokens),
        "byte_hit_rate": hit_bytes / max(1, requested_bytes),
        "matched_prefill_tokens": sum(int(row.get("matched_prefill_tokens", 0)) for row in rows),
        "avoided_prefill_tokens": sum(int(row.get("avoided_prefill_tokens", 0)) for row in rows),
        "cold_hit_rate": (
            statistics.fmean(hit_ratios[:cold_requests])
            if hit_ratios[:cold_requests] else math.nan
        ),
        "steady_state_hit_rate": (
            statistics.fmean(hit_ratios[cold_requests:])
            if hit_ratios[cold_requests:] else math.nan
        ),
        "ttft_mean_s": statistics.fmean(ttft) if ttft else math.nan,
        "ttft_p50_s": percentile(ttft, 0.50),
        "ttft_p95_s": percentile(ttft, 0.95),
        "accuracy": _accuracy(labels),
        "quality_hard_accuracy": _accuracy(hard),
        "insertions": max((int(row.get("insertions", 0)) for row in rows), default=0),
        "evictions": max((int(row.get("evictions", 0)) for row in rows), default=0),
        "prefill_time_s": sum(float(row.get("prefill_s", 0)) for row in rows),
        "cache_bytes_peak": max((int(row.get("cache_bytes", 0)) for row in rows), default=0),
        "useful_bytes_peak": max((int(row.get("useful_bytes", 0)) for row in rows), default=0),
        "shared_bytes_peak": max((int(row.get("shared_bytes", 0)) for row in rows), default=0),
        "stranded_bytes_peak": max((int(row.get("stranded_bytes", 0)) for row in rows), default=0),
        "metadata_bytes_peak": max(
            (int(row.get("metadata_bytes", 0)) for row in rows), default=0
        ),
        "cache_footprint_bytes_peak": max(
            (int(row.get("cache_footprint_bytes", row.get("cache_bytes", 0))) for row in rows),
            default=0,
        ),
        "cached_documents_peak": max((int(row.get("cached_documents", 0)) for row in rows), default=0),
        "cached_blocks_peak": max((int(row.get("cached_blocks", 0)) for row in rows), default=0),
        "radix_nodes_peak": max((int(row.get("radix_nodes", 0)) for row in rows), default=0),
        "root_nodes_peak": max((int(row.get("root_nodes", 0)) for row in rows), default=0),
        "document_tree_nodes_peak": max(
            (int(row.get("document_tree_nodes", 0)) for row in rows), default=0
        ),
        "cached_tokens_peak": max(
            (int(row.get("cached_tokens", 0)) for row in rows), default=0
        ),
        "occupancy_mean": (
            statistics.fmean(float(row.get("occupancy", 0)) for row in rows)
            if rows else math.nan
        ),
        "occupancy_peak": max((float(row.get("occupancy", 0)) for row in rows), default=0),
        "process_rss_bytes_peak": max(
            (int(row.get("process_rss_bytes", 0)) for row in rows), default=0
        ),
        "process_rss_delta_bytes_peak": max(
            (int(row.get("process_rss_delta_bytes", 0)) for row in rows), default=0
        ),
        "mps_allocated_bytes_peak": max(
            (int(row.get("mps_allocated_bytes", 0)) for row in rows), default=0
        ),
        "mps_driver_bytes_peak": max(
            (int(row.get("mps_driver_bytes", 0)) for row in rows), default=0
        ),
        "mps_allocated_delta_bytes_peak": max(
            (int(row.get("mps_allocated_delta_bytes", 0)) for row in rows), default=0
        ),
        "mps_driver_delta_bytes_peak": max(
            (int(row.get("mps_driver_delta_bytes", 0)) for row in rows), default=0
        ),
        "cuda_allocated_bytes_peak": max(
            (int(row.get("cuda_allocated_bytes", 0)) for row in rows), default=0
        ),
        "cuda_reserved_bytes_peak": max(
            (int(row.get("cuda_reserved_bytes", 0)) for row in rows), default=0
        ),
        "cuda_allocated_delta_bytes_peak": max(
            (int(row.get("cuda_allocated_delta_bytes", 0)) for row in rows), default=0
        ),
        "cuda_reserved_delta_bytes_peak": max(
            (int(row.get("cuda_reserved_delta_bytes", 0)) for row in rows), default=0
        ),
    }
    comparisons = [row for row in rows if row.get("reference_agreement") is not None]
    result["reference_label_agreement"] = (
        sum(bool(row["reference_agreement"]) for row in comparisons) / len(comparisons)
        if comparisons else math.nan
    )
    reference_labeled = [
        row for row in labels if row.get("fp16_reference_label") is not None
    ]
    reference_accuracy = (
        sum(row["fp16_reference_label"] == row["gold_label"] for row in reference_labeled)
        / len(reference_labeled)
        if reference_labeled else math.nan
    )
    result["fp16_reference_accuracy"] = reference_accuracy
    result["accuracy_delta_vs_fp16"] = (
        result["accuracy"] - reference_accuracy
        if not math.isnan(result["accuracy"]) and not math.isnan(reference_accuracy)
        else math.nan
    )
    result["amortized_prefill_time_s"] = result["prefill_time_s"] / max(1, len(rows))
    for name in ("lookup_s", "load_s", "transfer_s", "dequant_s", "policy_s"):
        result[f"{name[:-2]}_mean_s"] = statistics.fmean(
            float(row.get(name, 0)) for row in rows
        ) if rows else math.nan
    return result


def _accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return math.nan
    return sum(row["predicted_label"] == row["gold_label"] for row in rows) / len(rows)


def label_agreement(rows: Iterable[dict[str, Any]], left: str, right: str) -> float:
    pairs = [(row.get(left), row.get(right)) for row in rows]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    return sum(a == b for a, b in pairs) / max(1, len(pairs))


def group_summaries(rows: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in fields)].append(row)
    output = []
    for values, group in groups.items():
        record = dict(zip(fields, values))
        record.update(summarize(group, cold_requests=int(group[0].get("cold_requests", 0))))
        output.append(record)
    return output
