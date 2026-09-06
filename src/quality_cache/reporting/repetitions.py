"""Run-level timing comparisons, without treating requests as independent trials."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from ..schema import RESULT_SCHEMA_VERSION
from .io import write_csv
from .metrics import percentile, summarize


ANALYSIS_VERSION = "quality-strategy-repetitions-v1"
STRATEGIES = ("document", "fixed-block", "radix")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_paths(path: Path) -> tuple[Path, ...]:
    return (
        path,
        path.with_suffix(".summary.json"),
        path.with_suffix(".jsonl.manifest.json"),
    )


def _trace(rows: list[dict]) -> list[tuple]:
    # Zipf request IDs can repeat. Position, article, tokens and gold must agree.
    return [
        tuple(row.get(key) for key in (
            "request_index", "request_id", "article_id", "article_tokens",
            "total_prompt_tokens", "gold_label", "difficult",
        ))
        for row in rows
    ]


def read_repetition_run(
    path: Path, spec: dict, config: dict, profile: str,
) -> tuple[list[dict], dict]:
    """Validate one complete run; usable before creating a resume receipt."""
    for artifact in artifact_paths(path):
        if not artifact.is_file():
            raise ValueError(f"incomplete run: missing {artifact}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    manifest = json.loads(path.with_suffix(".jsonl.manifest.json").read_text())
    count = config["profiles"][profile]["limit"] or 2086
    if len(rows) != count or summary.get("requests") != count:
        raise ValueError(f"{path.name}: expected {count} requests")
    if [row.get("request_index") for row in rows] != list(range(count)):
        raise ValueError(f"{path.name}: non-contiguous request indexes")
    cached = spec["policy"] != "none"
    expected = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "execution_mode": "inference", "model": config["model"],
        "split": "dev", "seed": 42, "workload": spec["workload"],
        "policy": spec["policy"], "storage": "accelerator-fp16",
        "device": "cuda", "cache_strategy": spec["cache_strategy"] if cached else "none",
        "budget_bytes": config["budget_mb"] * 2**20 if cached else 0,
        "block_tokens": spec.get("block_tokens", config["block_tokens"]),
    }
    for key, value in expected.items():
        # Split is a summary/manifest field, not a per-request field.
        if summary.get(key) != value or (key != "split" and any(row.get(key) != value for row in rows)):
            raise ValueError(f"{path.name}: unexpected {key}, expected {value!r}")
    for row in rows:
        if not math.isfinite(float(row["ttft_s"])) or float(row["ttft_s"]) <= 0:
            raise ValueError(f"{path.name}: invalid measured TTFT")
        if not row.get("request_id") or not row.get("article_id"):
            raise ValueError(f"{path.name}: missing request/article identity")
        if row.get("gold_label") not in tuple("ABCD") or row.get("predicted_label") not in tuple("ABCD"):
            raise ValueError(f"{path.name}: invalid answer label")
        if not 0 <= row["cache_bytes"] <= expected["budget_bytes"]:
            raise ValueError(f"{path.name}: cache byte bound exceeded")
        if not 0 <= row["matched_prefix_tokens"] <= row["article_tokens"]:
            raise ValueError(f"{path.name}: invalid article reuse")
        if not cached and (
            row.get("baseline_mode") != "segmented"
            or row.get("inference_path") != "segmented-uncached"
            or row["matched_prefix_tokens"] != 0
        ):
            raise ValueError(f"{path.name}: expected a segmented no-article-cache control")
    manifest_expected = {
        **{key: value for key, value in expected.items() if key != "storage"},
        # Manifests record the CLI organization even for --policy none.
        "cache_strategy": spec["cache_strategy"],
        "storage_format": "accelerator-fp16", "dtype": "float16",
        "kv_backend": "tensor", "model_revision": config["model_revision"],
        "tokenizer_revision": config["model_revision"],
    }
    for key, value in manifest_expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{path.name}: incompatible manifest {key}")
    return rows, manifest


def _identity(manifest: dict) -> dict:
    fields = (
        "dataset_checksum", "model", "model_revision", "tokenizer_revision",
        "prompt_version", "result_schema_version", "git_revision", "timing_scope",
    )
    hardware_fields = (
        "python", "torch", "torch_cuda", "cuda_device_name",
        "cuda_device_total_memory", "cuda_device_capability",
    )
    identity = {key: manifest.get(key) for key in fields}
    identity["hardware"] = {
        key: manifest.get("hardware", {}).get(key) for key in hardware_fields
    }
    if any(value is None for value in identity.values()) or any(
        value is None for value in identity["hardware"].values()
    ):
        raise ValueError("missing code, dataset or runtime provenance")
    return identity


def analyze_repetitions(
    config: dict, results_dir: Path, output_dir: Path, *, profile: str = "confirmation",
    allow_incomplete: bool = False,
) -> dict[str, list[dict]]:
    """Summarize complete matched repetitions, with single correctness controls.

    Primary timing includes every request. A separately labelled sensitivity
    view drops the first 10% of each trace, without resetting cache state.
    There is deliberately no IID request bootstrap or significance declaration.
    """
    rows_by_name, inputs, identity, traces, groups = {}, [], None, {}, {}
    references, missing = {}, []
    for spec in config["runs"]:
        path = results_dir / f"dev_{profile}_{spec['name']}.jsonl"
        if allow_incomplete and not all(p.is_file() for p in artifact_paths(path)):
            missing.append(spec["name"])
            continue
        rows, manifest = read_repetition_run(path, spec, config, profile)
        current = _identity(manifest)
        if identity is not None and identity != current:
            raise ValueError(f"{path.name}: incompatible code/dataset/runtime manifests")
        identity = current
        workload = spec["workload"]
        trace = _trace(rows)
        if workload in traces and traces[workload] != trace:
            raise ValueError(f"{path.name}: unaligned {workload} trace")
        traces[workload] = trace
        strategy = spec["cache_strategy"] if spec["policy"] != "none" else "segmented"
        if strategy == "segmented":
            if workload in references:
                raise ValueError("expected only one correctness reference per workload")
            references[workload] = spec
        else:
            group = groups.setdefault((spec["repetition"], workload), {})
            if strategy in group:
                raise ValueError("duplicate strategy within a repetition/workload")
            group[strategy] = spec
        if spec.get("reference_run"):
            reference = spec["reference_run"]
            reference_path = results_dir / f"dev_{profile}_{reference}.jsonl"
            if reference not in rows_by_name or manifest.get("reference_checksum") != sha256(reference_path):
                raise ValueError(f"{path.name}: missing or changed offline reference")
            if _trace(rows_by_name[reference]) != trace:
                raise ValueError(f"{path.name}: unaligned offline reference")
        rows_by_name[spec["name"]] = rows
        inputs.append({
            "run": spec["name"], "repetition": spec["repetition"],
            "sha256": sha256(path),
            "manifest_sha256": sha256(path.with_suffix(".jsonl.manifest.json")),
        })

    run_metrics, paired, complete_groups = [], [], []
    for (repetition, workload), specs in groups.items():
        if set(specs) != set(STRATEGIES) or workload not in references:
            if allow_incomplete:
                continue
            raise ValueError("each repetition/workload needs all three caches and a reference")
        reference_spec = references[workload]
        for strategy in STRATEGIES:
            if specs[strategy].get("reference_run") != reference_spec["name"]:
                raise ValueError("each cache needs the matching workload's reference")
        complete_groups.append((repetition, workload))

    if not complete_groups:
        raise ValueError("no complete three-strategy repetition; finish a group before comparing")
    included = [
        spec for spec in config["runs"]
        if spec["name"] in rows_by_name and (
            spec["policy"] == "none" or (spec["repetition"], spec["workload"]) in complete_groups
        )
    ]
    for spec in included:
        reference = rows_by_name[references[spec["workload"]]["name"]]
        for scope in ("all_requests", "after_first_10_percent"):
            start = 0 if scope == "all_requests" else max(1, len(reference) // 10)
            rows = rows_by_name[spec["name"]][start:]
            ttft = [float(row["ttft_s"]) for row in rows]
            summary = summarize(rows)
            mismatches = sum(
                row["predicted_label"] != ref["predicted_label"]
                for row, ref in zip(rows, reference[start:])
            )
            run_metrics.append({
                "run": spec["name"], "repetition": spec["repetition"], "workload": spec["workload"],
                "strategy": spec["cache_strategy"] if spec["policy"] != "none" else "segmented",
                "scope": scope, "requests": len(rows), "ttft_mean_s": statistics.fmean(ttft),
                "ttft_p50_s": percentile(ttft, 0.5), "ttft_p90_s": percentile(ttft, 0.9),
                "reference_label_mismatches": mismatches,
                "reference_label_agreement": 1 - mismatches / len(rows),
                **{key: summary[key] for key in (
                    "article_token_hit_rate", "accuracy", "quality_hard_accuracy",
                    "policy_mean_s", "lookup_mean_s", "cache_bytes_peak",
                )},
            })

    for repetition, workload in complete_groups:
        for scope in ("all_requests", "after_first_10_percent"):
            scoped = {
                row["strategy"]: row for row in run_metrics
                if (row["repetition"], row["workload"], row["scope"]) == (repetition, workload, scope)
            }
            for baseline, candidate in itertools.combinations(STRATEGIES, 2):
                base, cache = scoped[baseline], scoped[candidate]
                paired.append({
                    "repetition": repetition, "workload": workload, "scope": scope,
                    "baseline": baseline, "candidate": candidate,
                    "mean_speedup": base["ttft_mean_s"] / cache["ttft_mean_s"],
                    "mean_reduction_percent": 100 * (1 - cache["ttft_mean_s"] / base["ttft_mean_s"]),
                    "p90_reduction_percent": 100 * (1 - cache["ttft_p90_s"] / base["ttft_p90_s"]),
                })

    def aggregate(source: list[dict], keys: tuple, metrics: tuple) -> list[dict]:
        buckets = {}
        for row in source:
            buckets.setdefault(tuple(row[key] for key in keys), []).append(row)
        output = []
        for key, rows in buckets.items():
            item = dict(zip(keys, key))
            item["repetitions"] = len(rows)
            for metric in metrics:
                values = [row[metric] for row in rows]
                for label, fn in (("median", statistics.median), ("min", min), ("max", max)):
                    item[f"{metric}_{label}"] = fn(values)
                item[f"{metric}_stdev"] = statistics.stdev(values) if len(values) > 1 else math.nan
            if "mean_reduction_percent" in metrics:
                item["candidate_faster_runs"] = sum(row["mean_reduction_percent"] > 0 for row in rows)
            output.append(item)
        return output

    artifacts = {
        "run_metrics": run_metrics,
        "strategy_variation": aggregate(run_metrics, ("workload", "strategy", "scope"), ("ttft_mean_s", "ttft_p90_s")),
        "paired_runs": paired,
        "paired_variation": aggregate(paired, ("workload", "scope", "baseline", "candidate"), ("mean_reduction_percent", "p90_reduction_percent")),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in artifacts.items():
        write_csv(output_dir / f"{name}.csv", rows)
    (output_dir / "analysis.json").write_text(json.dumps({
        "analysis_version": ANALYSIS_VERSION, "profile": profile,
        "identity": identity, "inputs": inputs, "missing_runs": missing,
        "complete_groups": complete_groups,
        "excluded_unpaired_runs": [name for name in rows_by_name if name not in {s["name"] for s in included}],
        "experimental_unit": "one fresh-process run, paired within repetition and workload",
        "reference_role": "one correctness reference per workload; not independent repeated timing controls",
        "scope_note": "First-10% exclusion is diagnostic; all-request timing is primary.",
        "limitations": "At most three repeats of seed 42 on one GPU model; descriptive ranges, not significance tests or workload generalization. Check coverage for incomplete groups.",
    }, indent=2) + "\n")
    return artifacts
