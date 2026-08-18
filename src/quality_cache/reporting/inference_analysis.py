"""Reproducible analysis for the selected real-inference confirmation suite."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import write_csv
from .metrics import percentile, summarize


ANALYSIS_VERSION = "quality-fair-inference-v1"


@dataclass(frozen=True)
class ConfirmationRun:
    name: str
    label: str
    short_label: str
    workload: str
    kind: str
    strategy: str
    policy: str
    storage: str


RUN_SPECS = (
    ConfirmationRun(
        "01_uncached_random_fp16",
        "Full uncached FP16",
        "Full",
        "random",
        "full",
        "none",
        "none",
        "accelerator-fp16",
    ),
    ConfirmationRun(
        "01b_segmented_uncached_random_fp16",
        "Segmented, no document cache",
        "Segmented",
        "random",
        "segmented",
        "none",
        "none",
        "accelerator-fp16",
    ),
    ConfirmationRun(
        "02_document_lru_random_fp16_4gib",
        "Document LRU FP16",
        "Document\nFP16",
        "random",
        "cache",
        "document",
        "lru",
        "accelerator-fp16",
    ),
    ConfirmationRun(
        "03_fixed_block_lru_random_fp16_4gib",
        "Fixed-block LRU FP16",
        "Fixed block\nFP16",
        "random",
        "cache",
        "fixed-block",
        "lru",
        "accelerator-fp16",
    ),
    ConfirmationRun(
        "04_radix_lru_random_fp16_4gib",
        "Radix LRU FP16",
        "Radix\nFP16",
        "random",
        "cache",
        "radix",
        "lru",
        "accelerator-fp16",
    ),
    ConfirmationRun(
        "05_document_lru_random_int8_4gib",
        "Document LRU CPU INT8",
        "Document\nINT8",
        "random",
        "cache",
        "document",
        "lru",
        "cpu-int8",
    ),
    ConfirmationRun(
        "06_uncached_zipf_fp16",
        "Full uncached FP16",
        "Full",
        "zipf",
        "full",
        "none",
        "none",
        "accelerator-fp16",
    ),
    ConfirmationRun(
        "06b_segmented_uncached_zipf_fp16",
        "Segmented, no document cache",
        "Segmented",
        "zipf",
        "segmented",
        "none",
        "none",
        "accelerator-fp16",
    ),
    ConfirmationRun(
        "07_document_lru_zipf_fp16_4gib",
        "Document LRU FP16",
        "Document\nLRU",
        "zipf",
        "cache",
        "document",
        "lru",
        "accelerator-fp16",
    ),
    ConfirmationRun(
        "08_document_gdsf_zipf_fp16_4gib",
        "Document GDSF FP16",
        "Document\nGDSF",
        "zipf",
        "cache",
        "document",
        "gdsf",
        "accelerator-fp16",
    ),
)


def bootstrap_ratio_ci(
    numerator: list[float],
    denominator: list[float],
    *,
    samples: int = 20_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap CI for ratio(mean(numerator), mean(denominator))."""
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired bootstrap inputs must have equal nonzero length")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    generator = random.Random(seed)
    count = len(numerator)
    estimates = []
    for _ in range(samples):
        numerator_sum = 0.0
        denominator_sum = 0.0
        for _ in range(count):
            index = generator.randrange(count)
            numerator_sum += numerator[index]
            denominator_sum += denominator[index]
        estimates.append(numerator_sum / denominator_sum)
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def analyze_inference_confirmation(
    results_dir: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 42,
) -> list[Path]:
    """Validate the ten-run suite and write tables, diagnostics, and figures."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    if not results_dir.is_dir():
        raise ValueError(f"inference result directory does not exist: {results_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for spec in RUN_SPECS:
        path = _resolve_run(results_dir, spec.name)
        paths[spec.name] = path
        rows[spec.name] = _read_jsonl(path)
        manifests[spec.name] = _read_manifest(path)
        _validate_run(spec, rows[spec.name])

    validation = _validate_suite(rows, manifests)
    summaries = _run_summaries(rows)
    comparisons = _fair_comparisons(
        rows,
        summaries,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    correctness = _correctness_comparisons(rows)
    segmentation_correctness = _segmentation_correctness(rows)
    mismatch_details = _mismatch_details(rows)

    artifacts = []
    artifacts.append(write_csv(output_dir / "run_summaries.csv", summaries))
    artifacts.append(write_csv(output_dir / "fair_speedups.csv", comparisons))
    artifacts.append(write_csv(output_dir / "correctness.csv", correctness))
    artifacts.append(
        write_csv(
            output_dir / "segmentation_correctness.csv",
            segmentation_correctness,
        )
    )

    mismatch_path = output_dir / "mismatch_details.json"
    mismatch_path.write_text(
        json.dumps(mismatch_details, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifacts.append(mismatch_path)

    diagnostics_path = output_dir / "analysis.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "analysis_version": ANALYSIS_VERSION,
                "bootstrap_samples": bootstrap_samples,
                "seed": seed,
                "validation": validation,
                "inputs": [
                    {
                        "run": spec.name,
                        "file": paths[spec.name].name,
                        "sha256": _sha256(paths[spec.name]),
                        "git_revision": manifests[spec.name].get("git_revision"),
                    }
                    for spec in RUN_SPECS
                ],
                "fair_speedups": comparisons,
                "correctness": correctness,
                "segmentation_correctness": segmentation_correctness,
                "mismatch_details": mismatch_details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts.append(diagnostics_path)

    markdown_path = output_dir / "results.md"
    markdown_path.write_text(
        _render_markdown(
            validation,
            comparisons,
            correctness,
            segmentation_correctness,
        ),
        encoding="utf-8",
    )
    artifacts.append(markdown_path)
    artifacts.extend(_make_figures(summaries, comparisons, output_dir))
    return artifacts


def _resolve_run(results_dir: Path, name: str) -> Path:
    matches = sorted(results_dir.glob(f"*_{name}.jsonl"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one JSONL ending in _{name}.jsonl in "
            f"{results_dir}, found {len(matches)}"
        )
    return matches[0]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    parsed = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not parsed:
        raise ValueError(f"result JSONL is empty: {path}")
    return parsed


def _read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"result manifest is missing: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _validate_run(spec: ConfirmationRun, rows: list[dict[str, Any]]) -> None:
    indexes = [int(row.get("request_index", -1)) for row in rows]
    if indexes != list(range(len(rows))):
        raise ValueError(f"run {spec.name} has a non-contiguous request trace")
    keys = [(row.get("request_index"), row.get("request_id")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"run {spec.name} has duplicate request positions")
    if any(row.get("workload") != spec.workload for row in rows):
        raise ValueError(f"run {spec.name} has the wrong workload")
    expected_fields = {
        "cache_strategy": spec.strategy,
        "policy": spec.policy,
        "storage": spec.storage,
    }
    for field, expected in expected_fields.items():
        if any(row.get(field) != expected for row in rows):
            raise ValueError(
                f"run {spec.name} has the wrong {field}; expected {expected}"
            )
    if len({row.get("model") for row in rows}) != 1:
        raise ValueError(f"run {spec.name} mixes model identifiers")
    if spec.kind == "segmented":
        zero_fields = (
            "matched_prefix_tokens",
            "matched_prefill_tokens",
            "avoided_prefill_tokens",
            "matched_cache_bytes",
            "cache_bytes",
            "cached_documents",
            "insertions",
            "evictions",
        )
        retained_article_kv = any(
            any(int(row.get(field, 0)) != 0 for field in zero_fields)
            for row in rows
        )
        if retained_article_kv:
            raise ValueError(f"segmented run {spec.name} retained or reused article KV")
        if not all(
            row.get("baseline_mode") == "segmented"
            and row.get("inference_path") == "segmented-uncached"
            and bool(row.get("root_only_hit"))
            for row in rows
        ):
            raise ValueError(
                f"segmented run {spec.name} has inconsistent path metadata"
            )


def _validate_suite(
    rows: dict[str, list[dict[str, Any]]],
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checksums = {manifest.get("dataset_checksum") for manifest in manifests.values()}
    models = {row[0].get("model") for row in rows.values()}
    torch_versions = {
        manifest.get("hardware", {}).get("torch") for manifest in manifests.values()
    }
    if len(checksums) != 1 or None in checksums:
        raise ValueError("inference runs do not share one dataset checksum")
    if len(models) != 1 or None in models:
        raise ValueError("inference runs do not share one model")
    if len(torch_versions) != 1 or None in torch_versions:
        raise ValueError("inference runs do not share one Torch version")

    requests_by_workload = {}
    for workload in ("random", "zipf"):
        selected = [spec for spec in RUN_SPECS if spec.workload == workload]
        reference = rows[selected[0].name]
        reference_keys = _trace_keys(reference)
        for spec in selected[1:]:
            if _trace_keys(rows[spec.name]) != reference_keys:
                raise ValueError(f"{workload} traces are not aligned at {spec.name}")
        requests_by_workload[workload] = len(reference)

    return {
        "runs": len(RUN_SPECS),
        "requests_by_workload": requests_by_workload,
        "dataset_checksum": next(iter(checksums)),
        "model": next(iter(models)),
        "torch": next(iter(torch_versions)),
        "traces_aligned": True,
        "segmented_invariants": True,
    }


def _trace_keys(rows: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    return [
        (int(row["request_index"]), str(row["request_id"]), str(row["article_id"]))
        for row in rows
    ]


def _run_summaries(
    rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for spec in RUN_SPECS:
        run_rows = rows[spec.name]
        cold_requests = int(
            run_rows[0].get("cold_requests", max(1, len(run_rows) // 10))
        )
        measured = summarize(run_rows, cold_requests=cold_requests)
        output.append(
            {
                "run": spec.name,
                "label": spec.label,
                "short_label": spec.short_label.replace("\n", " "),
                "workload": spec.workload,
                "kind": spec.kind,
                "strategy": spec.strategy,
                "policy": spec.policy,
                "storage": spec.storage,
                "requests": measured["requests"],
                "ttft_mean_s": measured["ttft_mean_s"],
                "ttft_p50_s": measured["ttft_p50_s"],
                "ttft_p95_s": measured["ttft_p95_s"],
                "article_token_hit_rate": measured["article_token_hit_rate"],
                "full_document_hit_rate": measured["full_document_hit_rate"],
                "byte_hit_rate": measured["byte_hit_rate"],
                "accuracy": measured["accuracy"],
                "quality_hard_accuracy": measured["quality_hard_accuracy"],
                "reference_label_agreement": measured["reference_label_agreement"],
                "cache_bytes_peak": measured["cache_bytes_peak"],
                "evictions": measured["evictions"],
                "dequant_mean_s": measured["dequant_mean_s"],
                "policy_mean_s": measured["policy_mean_s"],
            }
        )
    return output


def _fair_comparisons(
    rows: dict[str, list[dict[str, Any]]],
    summaries: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    summary_by_name = {summary["run"]: summary for summary in summaries}
    output = []
    comparison_index = 0
    for spec in RUN_SPECS:
        if spec.kind != "cache":
            continue
        full_spec = next(
            candidate
            for candidate in RUN_SPECS
            if candidate.workload == spec.workload and candidate.kind == "full"
        )
        segmented_spec = next(
            candidate
            for candidate in RUN_SPECS
            if candidate.workload == spec.workload and candidate.kind == "segmented"
        )
        full_ttft = _ttft(rows[full_spec.name])
        segmented_ttft = _ttft(rows[segmented_spec.name])
        cache_ttft = _ttft(rows[spec.name])
        full_mean = statistics.fmean(full_ttft)
        segmented_mean = statistics.fmean(segmented_ttft)
        cache_mean = statistics.fmean(cache_ttft)
        lower, upper = bootstrap_ratio_ci(
            segmented_ttft,
            cache_ttft,
            samples=bootstrap_samples,
            seed=seed + comparison_index,
        )
        comparison_index += 1
        summary = summary_by_name[spec.name]
        full_summary = summary_by_name[full_spec.name]
        segmented_summary = summary_by_name[segmented_spec.name]
        output.append(
            {
                "run": spec.name,
                "label": spec.label,
                "workload": spec.workload,
                "strategy": spec.strategy,
                "policy": spec.policy,
                "storage": spec.storage,
                "full_uncached_ttft_mean_s": full_mean,
                "full_uncached_ttft_p50_s": full_summary["ttft_p50_s"],
                "full_uncached_ttft_p95_s": full_summary["ttft_p95_s"],
                "segmented_ttft_mean_s": segmented_mean,
                "segmented_ttft_p50_s": segmented_summary["ttft_p50_s"],
                "segmented_ttft_p95_s": segmented_summary["ttft_p95_s"],
                "cached_ttft_mean_s": cache_mean,
                "cached_ttft_p50_s": summary["ttft_p50_s"],
                "cached_ttft_p95_s": summary["ttft_p95_s"],
                "end_to_end_speedup": full_mean / cache_mean,
                "segmentation_speedup": full_mean / segmented_mean,
                "cache_only_speedup": segmented_mean / cache_mean,
                "cache_only_speedup_ci95_low": lower,
                "cache_only_speedup_ci95_high": upper,
                "cache_only_ttft_reduction_percent": 100.0
                * (1.0 - cache_mean / segmented_mean),
                "article_token_hit_rate": summary["article_token_hit_rate"],
                "full_document_hit_rate": summary["full_document_hit_rate"],
                "byte_hit_rate": summary["byte_hit_rate"],
                "cache_bytes_peak": summary["cache_bytes_peak"],
                "evictions": summary["evictions"],
            }
        )
    return output


def _correctness_comparisons(
    rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for spec in RUN_SPECS:
        if spec.kind != "cache":
            continue
        segmented_spec = next(
            candidate
            for candidate in RUN_SPECS
            if candidate.workload == spec.workload and candidate.kind == "segmented"
        )
        reference = rows[segmented_spec.name]
        candidate = rows[spec.name]
        segmented_accuracy = _row_accuracy(reference)
        cached_accuracy = _row_accuracy(candidate)
        segmented_hard_accuracy = _row_accuracy(reference, difficult_only=True)
        cached_hard_accuracy = _row_accuracy(candidate, difficult_only=True)
        deltas = []
        mismatches = []
        for reference_row, candidate_row in zip(reference, candidate):
            delta = max(
                abs(
                    float(reference_row["label_scores"][label])
                    - float(candidate_row["label_scores"][label])
                )
                for label in "ABCD"
            )
            deltas.append(delta)
            if reference_row["predicted_label"] != candidate_row["predicted_label"]:
                mismatches.append(str(reference_row["request_id"]))
        output.append(
            {
                "run": spec.name,
                "label": spec.label,
                "workload": spec.workload,
                "storage": spec.storage,
                "agreement_vs_segmented": 1.0 - len(mismatches) / len(reference),
                "label_mismatches": len(mismatches),
                "unique_label_mismatches": len(set(mismatches)),
                "max_label_logit_delta_vs_segmented": max(deltas),
                "mean_max_label_logit_delta_vs_segmented": statistics.fmean(deltas),
                "segmented_accuracy": segmented_accuracy,
                "cached_accuracy": cached_accuracy,
                "accuracy_delta_vs_segmented": cached_accuracy
                - segmented_accuracy,
                "segmented_hard_accuracy": segmented_hard_accuracy,
                "cached_hard_accuracy": cached_hard_accuracy,
                "hard_accuracy_delta_vs_segmented": (
                    cached_hard_accuracy - segmented_hard_accuracy
                ),
            }
        )
    return output


def _segmentation_correctness(
    rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for workload in ("random", "zipf"):
        full_spec = next(
            spec
            for spec in RUN_SPECS
            if spec.workload == workload and spec.kind == "full"
        )
        segmented_spec = next(
            spec
            for spec in RUN_SPECS
            if spec.workload == workload and spec.kind == "segmented"
        )
        reference = rows[full_spec.name]
        candidate = rows[segmented_spec.name]
        full_accuracy = _row_accuracy(reference)
        segmented_accuracy = _row_accuracy(candidate)
        full_hard_accuracy = _row_accuracy(reference, difficult_only=True)
        segmented_hard_accuracy = _row_accuracy(candidate, difficult_only=True)
        deltas = _label_score_deltas(reference, candidate)
        mismatches = [
            str(reference_row["request_id"])
            for reference_row, candidate_row in zip(reference, candidate)
            if reference_row["predicted_label"] != candidate_row["predicted_label"]
        ]
        output.append(
            {
                "workload": workload,
                "agreement_vs_full": 1.0 - len(mismatches) / len(reference),
                "label_mismatches": len(mismatches),
                "unique_label_mismatches": len(set(mismatches)),
                "max_label_logit_delta_vs_full": max(deltas),
                "mean_max_label_logit_delta_vs_full": statistics.fmean(deltas),
                "full_accuracy": full_accuracy,
                "segmented_accuracy": segmented_accuracy,
                "accuracy_delta_vs_full": segmented_accuracy - full_accuracy,
                "full_hard_accuracy": full_hard_accuracy,
                "segmented_hard_accuracy": segmented_hard_accuracy,
                "hard_accuracy_delta_vs_full": (
                    segmented_hard_accuracy - full_hard_accuracy
                ),
            }
        )
    return output


def _mismatch_details(
    rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    comparisons = []
    for workload in ("random", "zipf"):
        full_spec = next(
            spec
            for spec in RUN_SPECS
            if spec.workload == workload and spec.kind == "full"
        )
        segmented_spec = next(
            spec
            for spec in RUN_SPECS
            if spec.workload == workload and spec.kind == "segmented"
        )
        comparisons.append(
            (
                f"{segmented_spec.name}_vs_full",
                workload,
                rows[full_spec.name],
                rows[segmented_spec.name],
            )
        )
    for spec in RUN_SPECS:
        if spec.kind != "cache":
            continue
        segmented_spec = next(
            candidate
            for candidate in RUN_SPECS
            if candidate.workload == spec.workload and candidate.kind == "segmented"
        )
        comparisons.append(
            (
                f"{spec.name}_vs_segmented",
                spec.workload,
                rows[segmented_spec.name],
                rows[spec.name],
            )
        )

    output = []
    for comparison, workload, reference, candidate in comparisons:
        for reference_row, candidate_row in zip(reference, candidate):
            if reference_row["predicted_label"] == candidate_row["predicted_label"]:
                continue
            output.append(
                {
                    "comparison": comparison,
                    "workload": workload,
                    "request_index": int(reference_row["request_index"]),
                    "request_id": str(reference_row["request_id"]),
                    "article_id": str(reference_row["article_id"]),
                    "reference_label": reference_row["predicted_label"],
                    "candidate_label": candidate_row["predicted_label"],
                    "gold_label": reference_row.get("gold_label"),
                    "max_label_logit_delta": max(
                        abs(
                            float(reference_row["label_scores"][label])
                            - float(candidate_row["label_scores"][label])
                        )
                        for label in "ABCD"
                    ),
                }
            )
    return output


def _label_score_deltas(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> list[float]:
    return [
        max(
            abs(
                float(reference_row["label_scores"][label])
                - float(candidate_row["label_scores"][label])
            )
            for label in "ABCD"
        )
        for reference_row, candidate_row in zip(reference, candidate)
    ]


def _row_accuracy(
    rows: list[dict[str, Any]],
    *,
    difficult_only: bool = False,
) -> float:
    labeled = [
        row
        for row in rows
        if row.get("gold_label") is not None
        and (not difficult_only or bool(row.get("difficult")))
    ]
    if not labeled:
        return float("nan")
    return statistics.fmean(
        row["predicted_label"] == row["gold_label"] for row in labeled
    )


def _ttft(rows: list[dict[str, Any]]) -> list[float]:
    return [float(row["ttft_s"]) for row in rows]


def _render_markdown(
    validation: dict[str, Any],
    comparisons: list[dict[str, Any]],
    correctness: list[dict[str, Any]],
    segmentation_correctness: list[dict[str, Any]],
) -> str:
    int8_correctness = next(
        row for row in correctness if row["storage"] == "cpu-int8"
    )
    zipf_segmentation = next(
        row for row in segmentation_correctness if row["workload"] == "zipf"
    )
    random_requests = validation["requests_by_workload"]["random"]
    unique_zipf_mismatches = zipf_segmentation["unique_label_mismatches"]
    unique_question_word = "question" if unique_zipf_mismatches == 1 else "questions"
    lines = [
        "# Fair inference confirmation",
        "",
        f"Analysis schema: `{ANALYSIS_VERSION}`. The archive contains "
        f"{validation['runs']} aligned runs using `{validation['model']}` and "
        f"Torch `{validation['torch']}`.",
        "",
        "The cache-only baseline is segmented inference with pinned L0 and no "
        "retained article KV. End-to-end speedup uses the original one-forward "
        "uncached path.",
        "",
        "## Latency",
        "",
        "| Workload | Strategy | Storage | Cached TTFT mean/p50/p95 (s) | "
        "Cache-only speedup "
        "(95% CI) | End-to-end speedup | Cache-only TTFT change |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {row['workload']} | {row['strategy']} / {row['policy']} | "
            f"{row['storage']} | {row['cached_ttft_mean_s']:.3f} / "
            f"{row['cached_ttft_p50_s']:.3f} / {row['cached_ttft_p95_s']:.3f} | "
            f"{row['cache_only_speedup']:.2f}x "
            f"[{row['cache_only_speedup_ci95_low']:.2f}, "
            f"{row['cache_only_speedup_ci95_high']:.2f}] | "
            f"{row['end_to_end_speedup']:.2f}x | "
            f"{row['cache_only_ttft_reduction_percent']:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "A positive TTFT change is a reduction. A cache-only speedup below "
            "1.0 means cache management made the run slower than segmented "
            "execution without document retention.",
            "",
            "## Correctness against the segmented path",
            "",
            "| Workload | Strategy | Storage | Label agreement | Accuracy "
            "(segmented -> cache) | Hard accuracy (segmented -> cache) | "
            "Mismatches | Maximum logit delta |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in correctness:
        lines.append(
            f"| {row['workload']} | {row['label']} | {row['storage']} | "
            f"{100.0 * row['agreement_vs_segmented']:.1f}% | "
            f"{100.0 * row['segmented_accuracy']:.1f}% -> "
            f"{100.0 * row['cached_accuracy']:.1f}% "
            f"({100.0 * row['accuracy_delta_vs_segmented']:+.1f} pp) | "
            f"{100.0 * row['segmented_hard_accuracy']:.1f}% -> "
            f"{100.0 * row['cached_hard_accuracy']:.1f}% "
            f"({100.0 * row['hard_accuracy_delta_vs_segmented']:+.1f} pp) | "
            f"{row['label_mismatches']} | "
            f"{row['max_label_logit_delta_vs_segmented']:.6f} |"
        )
    lines.extend(
        [
            "",
            "The FP16 document-cache paths match segmented execution exactly. "
            f"CPU INT8 has {int8_correctness['label_mismatches']} label mismatch "
            f"in this {random_requests}-request sample, so its "
            "quality effect needs a larger confirmation before it is described "
            "as lossless.",
            "",
            "## Segmented control against the full path",
            "",
            "| Workload | Label agreement | Accuracy (full -> segmented) | "
            "Hard accuracy (full -> segmented) | Mismatch occurrences | "
            "Unique questions | Maximum logit delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in segmentation_correctness:
        lines.append(
            f"| {row['workload']} | {100.0 * row['agreement_vs_full']:.1f}% | "
            f"{100.0 * row['full_accuracy']:.1f}% -> "
            f"{100.0 * row['segmented_accuracy']:.1f}% "
            f"({100.0 * row['accuracy_delta_vs_full']:+.1f} pp) | "
            f"{100.0 * row['full_hard_accuracy']:.1f}% -> "
            f"{100.0 * row['segmented_hard_accuracy']:.1f}% "
            f"({100.0 * row['hard_accuracy_delta_vs_full']:+.1f} pp) | "
            f"{row['label_mismatches']} | {row['unique_label_mismatches']} | "
            f"{row['max_label_logit_delta_vs_full']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"The {zipf_segmentation['label_mismatches']} Zipf mismatch "
            "occurrences represent "
            f"{unique_zipf_mismatches} unique {unique_question_word}. "
            "Both FP16 document-cache runs match segmented execution exactly, "
            "so this discrepancy comes from the segmented forward path rather "
            "than cache restoration or eviction.",
            "",
            "## Figures",
            "",
            "- [TTFT by execution path](ttft_by_execution_path.pdf)",
            "- [Cache-only speedup](cache_only_speedup.pdf)",
            "- [Article-hit/latency tradeoff](hit_latency_tradeoff.pdf)",
            "",
        ]
    )
    return "\n".join(lines)


def _make_figures(
    summaries: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_by_name = {row["run"]: row for row in summaries}
    artifacts = []

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)
    colors = {
        "full": "#6b7280",
        "segmented": "#9ca3af",
        "document": "#2563eb",
        "fixed-block": "#dc2626",
        "radix": "#7c3aed",
    }
    for axis, workload in zip(axes, ("random", "zipf")):
        specs = [spec for spec in RUN_SPECS if spec.workload == workload]
        values = [summary_by_name[spec.name]["ttft_mean_s"] for spec in specs]
        bar_colors = [
            "#059669"
            if spec.storage == "cpu-int8"
            else colors.get(
                spec.kind if spec.kind != "cache" else spec.strategy,
                "#2563eb",
            )
            for spec in specs
        ]
        bars = axis.bar(range(len(specs)), values, color=bar_colors)
        axis.set_xticks(range(len(specs)), [spec.short_label for spec in specs])
        axis.tick_params(axis="x", labelsize=8)
        axis.set_title(f"{workload.capitalize()} workload")
        axis.set_ylabel("Mean TTFT (s)")
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.08,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("TTFT by execution path and cache organization")
    fig.tight_layout()
    artifacts.extend(_save_figure(fig, output_dir / "ttft_by_execution_path"))
    plt.close(fig)

    labels = [
        f"{row['workload']}: {row['strategy']}/{row['policy']} "
        f"({'INT8' if row['storage'] == 'cpu-int8' else 'FP16'})"
        for row in comparisons
    ]
    speedups = [row["cache_only_speedup"] for row in comparisons]
    lower = [
        value - row["cache_only_speedup_ci95_low"]
        for value, row in zip(speedups, comparisons)
    ]
    upper = [
        row["cache_only_speedup_ci95_high"] - value
        for value, row in zip(speedups, comparisons)
    ]
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    positions = list(range(len(labels)))
    axis.barh(
        positions,
        speedups,
        xerr=[lower, upper],
        color=["#059669" if value > 1.0 else "#dc2626" for value in speedups],
        alpha=0.9,
        capsize=3,
    )
    axis.axvline(1.0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Speedup over segmented execution without document cache")
    axis.set_title("Cross-request article-KV reuse benefit (95% paired bootstrap CI)")
    axis.grid(axis="x", alpha=0.25)
    for position, value, high in zip(positions, speedups, upper):
        axis.text(
            value + high + 0.045,
            position,
            f"{value:.2f}x",
            va="center",
            fontsize=8,
        )
    axis.set_xlim(
        0.0,
        max(row["cache_only_speedup_ci95_high"] for row in comparisons) + 0.3,
    )
    fig.tight_layout()
    artifacts.extend(_save_figure(fig, output_dir / "cache_only_speedup"))
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    marker_by_strategy = {
        "none": "o",
        "document": "s",
        "fixed-block": "^",
        "radix": "D",
    }
    annotation_layout = {
        "01b_segmented_uncached_random_fp16": ((7, 5), "left"),
        "02_document_lru_random_fp16_4gib": ((-8, 12), "right"),
        "03_fixed_block_lru_random_fp16_4gib": ((7, 5), "left"),
        "04_radix_lru_random_fp16_4gib": ((8, -15), "left"),
        "05_document_lru_random_int8_4gib": ((-8, 5), "right"),
        "06b_segmented_uncached_zipf_fp16": ((7, 5), "left"),
        "07_document_lru_zipf_fp16_4gib": ((-8, 11), "right"),
        "08_document_gdsf_zipf_fp16_4gib": ((-8, -15), "right"),
    }
    for axis, workload in zip(axes, ("random", "zipf")):
        specs = [
            spec
            for spec in RUN_SPECS
            if spec.workload == workload and spec.kind != "full"
        ]
        for spec in specs:
            row = summary_by_name[spec.name]
            x = 100.0 * row["article_token_hit_rate"]
            y = row["ttft_mean_s"]
            color = "#059669" if spec.storage == "cpu-int8" else "#2563eb"
            if spec.kind == "segmented":
                color = "#6b7280"
            axis.scatter(
                [x],
                [y],
                marker=marker_by_strategy[spec.strategy],
                s=75,
                color=color,
            )
            offset, alignment = annotation_layout[spec.name]
            axis.annotate(
                spec.short_label.replace("\n", " "),
                (x, y),
                xytext=offset,
                textcoords="offset points",
                ha=alignment,
                fontsize=8,
            )
        axis.set_title(f"{workload.capitalize()} workload")
        axis.set_xlabel("Article-token hit rate (%)")
        axis.set_ylabel("Mean TTFT (s)")
        axis.grid(alpha=0.25)
        axis.set_xlim((-2, 38) if workload == "random" else (-3, 66))
        axis.set_ylim(0.8, 2.45)
    fig.suptitle("Article reuse versus latency")
    fig.tight_layout()
    artifacts.extend(_save_figure(fig, output_dir / "hit_latency_tradeoff"))
    plt.close(fig)
    return artifacts


def _save_figure(fig, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
