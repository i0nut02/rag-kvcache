from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from .costs import load_prefill_cost_model
from .caches import CACHE_STRATEGIES
from .data import (
    build_workload,
    dataset_checksum,
    flatten_requests,
    grouped_max_hit_rate,
    load_quality_split,
)
from .inference.tensors import STORAGE_MODES
from .inference.reference import attach_offline_reference, load_reference_jsonl
from .prompt import PROMPT_VERSION
from .matrix import add_matrix_parser, run_matrix
from .reporting import make_primary_figures, summarize, write_csv, write_jsonl, write_manifest
from .schema import PREFILL_CALIBRATION_SCHEMA_VERSION, RESULT_SCHEMA_VERSION
from .simulation import (
    article_sizes,
    farthest_next_use_rows,
    simulate_trace,
    working_set_bytes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QuALITY article-prefix KV-cache experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-data", help="deduplicate and validate an official split")
    _dataset_args(validate)

    calibrate = sub.add_parser(
        "calibrate-prefill",
        help="measure MPS/CPU article-prefill costs for no-inference GDSF traces",
    )
    _dataset_args(calibrate)
    calibrate.add_argument("--model", required=True)
    calibrate.add_argument("--model-revision")
    calibrate.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    calibrate.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    calibrate.add_argument("--samples", type=_positive_int, default=5)
    calibrate.add_argument("--repeats", type=_positive_int, default=2)
    calibrate.add_argument("--warmup", type=int, default=1)
    calibrate.add_argument("--output", type=Path, required=True)

    simulate = sub.add_parser("simulate", help="run fast cache traces without model inference")
    _dataset_args(simulate)
    simulate.add_argument(
        "--workloads", nargs="+",
        choices=["grouped", "random", "zipf"],
        default=["grouped", "random", "zipf"],
    )
    simulate.add_argument(
        "--policies", nargs="+",
        choices=["lru", "lfu", "fifo", "gdsf", "farthest-next-use"],
        default=["lru", "lfu", "fifo", "gdsf", "farthest-next-use"],
    )
    simulate.add_argument("--seeds", nargs="+", type=int, default=[42])
    simulate_budget = simulate.add_mutually_exclusive_group()
    simulate_budget.add_argument("--budget-percent", nargs="+", type=float)
    simulate_budget.add_argument("--budget-mb", nargs="+", type=float)
    simulate.add_argument(
        "--requests",
        "--limit",
        dest="requests",
        type=_positive_int,
        metavar="N",
        help="limit the trace to N queries (default: all)",
    )
    simulate.add_argument(
        "--cold-requests", type=int,
        help="cold-start prefix length (default: first 10%% of each trace)",
    )
    simulate.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    simulate.add_argument("--model-revision")
    simulate.add_argument(
        "--tokenizer",
        help="optional local tokenizer path while retaining the canonical --model identity",
    )
    simulate.add_argument(
        "--tokenizer-mode", choices=["model", "approximate"], default="model",
        help="use exact model tokenization (default) or dependency-free trace estimates",
    )
    simulate.add_argument("--storage", choices=list(STORAGE_MODES), default="cpu-fp16")
    simulate.add_argument("--block-tokens", type=int, default=256)
    simulate.add_argument("--offline-prefill", action="store_true")
    simulate.add_argument("--prefill-calibration", type=Path)
    simulate.add_argument("--save-requests", action="store_true")
    simulate.add_argument("--output", type=Path, required=True, help="summary CSV path")

    run = sub.add_parser("run", help="run cache experiments, with inference by default")
    _dataset_args(run)
    run.add_argument(
        "--workload", choices=["grouped", "random", "zipf"], default="grouped"
    )
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--requests",
        "--limit",
        dest="requests",
        type=_positive_int,
        metavar="N",
        help="limit the run to N queries (default: all)",
    )
    run.add_argument(
        "--cold-requests", type=int,
        help="cold-start prefix length (default: first 10%% of the trace)",
    )
    run.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    run.add_argument("--model-revision")
    run.add_argument(
        "--tokenizer",
        help="optional local tokenizer/config path for --no-inference",
    )
    run.add_argument(
        "--no-inference",
        action="store_true",
        help="load tokenizer/config only and simulate cache behavior without model weights",
    )
    run.add_argument("--device", choices=["cpu", "mps", "cuda"], default="mps")
    run.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    run.add_argument(
        "--storage", choices=list(STORAGE_MODES), default="accelerator-fp16"
    )
    run.add_argument("--policy", choices=["none", "lru", "lfu", "fifo", "gdsf"], default="lru")
    run.add_argument(
        "--baseline-mode",
        choices=["full", "segmented"],
        default="full",
        help=(
            "uncached inference path used with --policy none: 'full' executes "
            "the complete prompt in one forward, while 'segmented' reuses only "
            "the pinned L0 root and recomputes the article and question without "
            "retaining article KV"
        ),
    )
    run.add_argument(
        "--cache-strategy", choices=list(CACHE_STRATEGIES), default="document"
    )
    budget = run.add_mutually_exclusive_group()
    budget.add_argument("--budget-mb", type=float)
    budget.add_argument(
        "--budget-percent",
        type=float,
        help="percentage of the full split FP16 article-KV working set",
    )
    run.add_argument("--max-articles", type=int)
    run.add_argument("--block-tokens", type=int, default=16)
    run.add_argument("--offline-prefill", action="store_true")
    run.add_argument(
        "--prefill-calibration",
        type=Path,
        help="measured prefill-cost JSON used by --no-inference GDSF",
    )
    run.add_argument("--validate-agreement", action="store_true")
    run.add_argument(
        "--reference-jsonl",
        type=Path,
        help=(
            "reuse a matching uncached FP16 JSONL for agreement metrics instead "
            "of executing an extra reference forward"
        ),
    )
    run.add_argument(
        "--strict-reference",
        action="store_true",
        help="fail an FP16 offline-reference comparison on a label or tolerance mismatch",
    )
    run.add_argument(
        "--agreement-atol",
        type=_nonnegative_float,
        help=(
            "cached/uncached label-logit absolute tolerance; default is dtype-aware "
            "(FP16 0.0625, BF16 0.25, FP32 0.001)"
        ),
    )
    run.add_argument(
        "--progress-every",
        type=_nonnegative_int,
        default=100,
        metavar="N",
        help="print serving progress every N requests; use 0 to disable (default: 100)",
    )
    run.add_argument("--output", type=Path, required=True, help="per-request JSONL path")

    plot = sub.add_parser("plot", help="create the four primary figures")
    plot.add_argument("summary_csv", type=Path)
    plot.add_argument("output_dir", type=Path)
    collect = sub.add_parser("collect", help="combine simulation/inference summary CSV files")
    collect.add_argument("inputs", nargs="+", type=Path)
    collect.add_argument("--output", type=Path, required=True)
    add_matrix_parser(sub)
    return parser


def _dataset_args(parser):
    parser.add_argument("dataset", type=Path, help="QuALITY.v1.0.1.htmlstripped SPLIT file")
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--verify-counts", action="store_true")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plot":
        for path in make_primary_figures(args.summary_csv, args.output_dir):
            print(path)
        return 0
    if args.command == "collect":
        rows = []
        for path in args.inputs:
            with path.open(encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        versions = {row.get("result_schema_version") for row in rows}
        if versions != {RESULT_SCHEMA_VERSION}:
            raise ValueError(
                "refusing to combine legacy or mixed result schemas: "
                f"found {sorted(str(value) for value in versions)}, "
                f"expected only {RESULT_SCHEMA_VERSION!r}"
            )
        write_csv(args.output, rows)
        write_manifest(
            args.output,
            {
                "run_type": "collection",
                "result_schema_version": RESULT_SCHEMA_VERSION,
                "inputs": [str(path) for path in args.inputs],
            },
        )
        print(f"wrote {len(rows)} summaries to {args.output}")
        return 0
    if args.command == "matrix":
        return run_matrix(args, main)
    articles = load_quality_split(
        args.dataset, split=args.split, verify_official_counts=args.verify_counts
    )
    if args.command == "validate-data":
        questions = sum(len(article.questions) for article in articles)
        print(json.dumps({
            "split": args.split,
            "articles": len(articles),
            "questions": questions,
            "grouped_max_hit_rate": grouped_max_hit_rate(articles),
            "sha256": dataset_checksum(args.dataset),
        }, indent=2))
        return 0
    if args.command == "calibrate-prefill":
        return _calibrate_prefill(args, articles)
    return _simulate(args, articles) if args.command == "simulate" else _run(args, articles)


def _calibrate_prefill(args, articles) -> int:
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    from .inference.model import QualityModelRunner

    runner = QualityModelRunner(
        args.model,
        device=args.device,
        dtype=args.dtype,
        revision=args.model_revision,
    )
    first_requests = list(
        {request.article_id: request for request in flatten_requests(articles)}.values()
    )
    ranked = sorted(
        ((runner.article_token_count(request), request) for request in first_requests),
        key=lambda item: (item[0], item[1].article_id),
    )
    sample_count = min(args.samples, len(ranked))
    if sample_count == 1:
        selected = [ranked[len(ranked) // 2]]
    else:
        # Avoid pathological minimum/maximum documents while covering the
        # central 80% of QuALITY article lengths.
        indices = [
            round(
                (0.1 + 0.8 * index / (sample_count - 1)) * (len(ranked) - 1)
            )
            for index in range(sample_count)
        ]
        selected = [ranked[index] for index in indices]

    warmup_request = selected[0][1]
    print(
        f"warming up with article {warmup_request.article_id} "
        f"({selected[0][0]} tokens)",
        flush=True,
    )
    for _ in range(args.warmup):
        runner.measure_article_prefill(warmup_request)

    samples = []
    for expected_tokens, request in selected:
        print(
            f"measuring article {request.article_id} ({expected_tokens} tokens)",
            flush=True,
        )
        measurements = []
        for _ in range(args.repeats):
            measured_tokens, elapsed = runner.measure_article_prefill(request)
            if measured_tokens != expected_tokens:
                raise AssertionError("article token count changed during calibration")
            measurements.append(elapsed)
        samples.append(
            {
                "article_id": request.article_id,
                "tokens": expected_tokens,
                "seconds": statistics.median(measurements),
                "measurements_s": measurements,
            }
        )
        print(
            f"calibrated article {request.article_id}: {expected_tokens} tokens, "
            f"median {samples[-1]['seconds']:.6f}s",
            flush=True,
        )

    payload = {
        "calibration_schema_version": PREFILL_CALIBRATION_SCHEMA_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "dataset": str(args.dataset),
        "dataset_checksum": dataset_checksum(args.dataset),
        "split": args.split,
        "model": args.model,
        "model_revision": runner.model_revision,
        "tokenizer_revision": runner.tokenizer_revision,
        "prompt_version": PROMPT_VERSION,
        "device": args.device,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(
        args.output,
        {
            "run_type": "prefill-calibration",
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "calibration_schema_version": PREFILL_CALIBRATION_SCHEMA_VERSION,
            "dataset": str(args.dataset),
            "dataset_checksum": dataset_checksum(args.dataset),
            "split": args.split,
            "model": args.model,
            "model_revision": runner.model_revision,
            "tokenizer_revision": runner.tokenizer_revision,
            "prompt_version": PROMPT_VERSION,
            "device": args.device,
            "dtype": args.dtype,
        },
    )
    print(f"wrote prefill calibration to {args.output}")
    return 0


def _simulate(args, articles) -> int:
    summaries = []
    request_rows = []
    prefill_cost_model = load_prefill_cost_model(args.prefill_calibration)
    tokenizer = None
    if args.tokenizer_mode == "model":
        from transformers import AutoTokenizer

        tokenizer_source = args.tokenizer or args.model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=None if args.tokenizer else args.model_revision,
            local_files_only=bool(args.tokenizer),
        )
        args.resolved_tokenizer_revision = str(
            getattr(tokenizer, "_commit_hash", None)
            or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
            or args.model_revision
            or tokenizer_source
        )
    else:
        args.resolved_tokenizer_revision = "approximate-regex-v1"
    corpus_trace = flatten_requests(articles)
    corpus_fp16_sizes, _ = article_sizes(
        corpus_trace,
        args.model,
        "cpu-fp16",
        tokenizer,
        args.block_tokens,
    )
    corpus_working_set = working_set_bytes(corpus_fp16_sizes)
    if args.budget_percent:
        budget_specs = [
            (percent, None, int(corpus_working_set * percent / 100))
            for percent in args.budget_percent
        ]
    else:
        budget_specs = [
            (None, mb, int(mb * 2**20))
            for mb in (args.budget_mb or [2048.0, 4096.0, 8192.0])
        ]
    for workload in args.workloads:
        seeds = [args.seeds[0]] if workload == "grouped" else args.seeds
        for seed in seeds:
            trace = build_workload(articles, workload, seed=seed, requests=args.requests)
            cold_requests = (
                args.cold_requests
                if args.cold_requests is not None
                else max(1, len(trace) // 10)
            )
            sizes, tokens = article_sizes(
                trace, args.model, args.storage, tokenizer, args.block_tokens
            )
            working_set = corpus_working_set
            for percent, budget_mb, budget in budget_specs:
                if budget < min(sizes.values()):
                    label = f"{percent}%" if percent is not None else f"{budget_mb} MiB"
                    raise ValueError(
                        f"{label} budget ({budget} bytes) cannot hold the smallest "
                        f"article ({min(sizes.values())} bytes)"
                    )
                for policy in args.policies:
                    started = time.perf_counter()
                    canonical_policy = policy
                    if canonical_policy == "farthest-next-use":
                        rows = farthest_next_use_rows(
                            trace, budget, args.model, args.storage,
                            tokenizer, args.block_tokens,
                            article_size_map=sizes,
                            article_token_counts=tokens,
                            prefill_cost_model=prefill_cost_model,
                        )
                    else:
                        rows = simulate_trace(
                            trace,
                            policy=canonical_policy,
                            budget_bytes=budget,
                            model=args.model,
                            storage=args.storage,
                            cold_requests=cold_requests,
                            offline_prefill=args.offline_prefill,
                            tokenizer=tokenizer,
                            block_tokens=args.block_tokens,
                            article_size_map=sizes,
                            article_token_counts=tokens,
                            prefill_cost_model=prefill_cost_model,
                        )
                    summary = summarize(rows, cold_requests=cold_requests)
                    offline_prefill_s = float(rows[0].get("offline_prefill_s", 0)) if rows else 0.0
                    summary.update({
                        "split": args.split,
                        "workload": workload,
                        "seed": seed,
                        "policy": canonical_policy,
                        "storage": args.storage,
                        "model": args.model,
                        "budget_percent": percent,
                        "budget_mb": budget_mb,
                        "budget_bytes": budget,
                        "working_set_bytes": working_set,
                        "simulation_wall_s": time.perf_counter() - started,
                        "cache_bytes": max((row.get("cache_bytes", 0) for row in rows), default=0),
                        "offline_prefill_s": offline_prefill_s,
                        "offline_amortized_prefill_s": offline_prefill_s / max(1, len(rows)),
                        "prefill_cost_model": prefill_cost_model.source,
                    })
                    summaries.append(summary)
                    if args.save_requests:
                        for row in rows:
                            row.update({
                                "workload": workload,
                                "seed": seed,
                                "budget_percent": percent,
                                "budget_mb": budget_mb,
                            })
                        request_rows.extend(rows)
    write_csv(args.output, summaries)
    if args.save_requests:
        write_jsonl(args.output.with_suffix(".requests.jsonl"), request_rows)
    write_manifest(args.output, _manifest(args, "simulation"))
    print(f"wrote {len(summaries)} summaries to {args.output}")
    return 0


def _run(args, articles) -> int:
    if args.storage == "accelerator-fp16" and args.device not in {"cuda", "mps"}:
        raise ValueError("accelerator-fp16 storage requires --device cuda or mps")
    if args.no_inference and args.validate_agreement:
        raise ValueError("--validate-agreement requires inference")
    if args.reference_jsonl is not None and args.no_inference:
        raise ValueError("--reference-jsonl requires inference")
    if args.reference_jsonl is not None and args.validate_agreement:
        raise ValueError(
            "--reference-jsonl and --validate-agreement are mutually exclusive"
        )
    if args.baseline_mode == "segmented" and args.policy != "none":
        raise ValueError("--baseline-mode segmented requires --policy none")
    if args.baseline_mode == "segmented" and args.no_inference:
        raise ValueError("--baseline-mode segmented requires inference")
    if (
        args.reference_jsonl is not None
        and args.policy == "none"
        and args.baseline_mode != "segmented"
    ):
        raise ValueError(
            "--reference-jsonl with --policy none requires "
            "--baseline-mode segmented"
        )
    if args.strict_reference and args.reference_jsonl is None:
        raise ValueError("--strict-reference requires --reference-jsonl")
    if not args.no_inference and args.prefill_calibration is not None:
        raise ValueError("--prefill-calibration is only used with --no-inference")
    if args.no_inference and args.offline_prefill:
        raise ValueError("--offline-prefill is not supported with --no-inference")
    if args.cache_strategy != "document" and args.max_articles is not None:
        raise ValueError("--max-articles is only supported by the document strategy")

    reference_rows = (
        load_reference_jsonl(
            args.reference_jsonl,
            expected_model=args.model,
            expected_workload=args.workload,
            expected_seed=args.seed,
        )
        if args.reference_jsonl is not None
        else None
    )

    if args.no_inference:
        from .inference.no_inference import NoInferenceRunner

        runner = NoInferenceRunner(
            args.model,
            tokenizer_source=args.tokenizer,
            device=args.device,
            dtype=args.dtype,
            revision=args.model_revision,
            prefill_calibration=args.prefill_calibration,
        )
    else:
        from .inference.model import QualityModelRunner

        runner = QualityModelRunner(
            args.model, device=args.device, dtype=args.dtype, revision=args.model_revision
        )
    args.resolved_agreement_atol = (
        args.agreement_atol
        if args.agreement_atol is not None
        else getattr(runner, "reference_logit_atol", None)
    )

    trace = build_workload(articles, args.workload, seed=args.seed, requests=args.requests)
    cold_requests = (
        args.cold_requests
        if args.cold_requests is not None
        else max(1, len(trace) // 10)
    )
    corpus_working_set = None
    if args.budget_percent is not None:
        first_requests = {
            request.article_id: request for request in flatten_requests(articles)
        }
        corpus_working_set = sum(
            runner.estimate_article_bytes(
                request, "cpu-fp16", args.block_tokens, args.cache_strategy
            )
            for request in first_requests.values()
        )
        budget_bytes = int(corpus_working_set * args.budget_percent / 100)
    else:
        budget_bytes = int((args.budget_mb or 0) * 2**20)
    args.resolved_budget_bytes = budget_bytes
    cache = None
    if args.policy != "none":
        if budget_bytes <= 0:
            raise ValueError("--budget-mb must be positive when caching is enabled")
        cache = runner.new_cache(
            budget_bytes,
            args.policy,
            args.max_articles,
            strategy=args.cache_strategy,
            block_tokens=args.block_tokens,
        )
    elif args.offline_prefill:
        raise ValueError("offline prefill is not applicable to --policy none")
    offline_prefill_s = 0.0
    if args.offline_prefill:
        counts = Counter(request.article_id for request in trace)
        first = {request.article_id: request for request in trace}
        for article_id, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            request = first[article_id]
            estimated = runner.estimate_article_bytes(
                request, args.storage, args.block_tokens, args.cache_strategy
            )
            count_full = args.max_articles is not None and len(cache) >= args.max_articles
            if count_full or cache.current_bytes + estimated > cache.max_bytes:
                continue
            offline_prefill_s += runner.prefill_article(
                request,
                cache,
                storage=args.storage,
                block_tokens=args.block_tokens,
            )

    rows = []
    serving_started = time.perf_counter()
    for index, request in enumerate(trace):
        if cache is None and args.baseline_mode == "segmented":
            row = runner.serve_segmented_uncached(
                request,
                storage=args.storage,
                block_tokens=args.block_tokens,
                cache_strategy=args.cache_strategy,
            )
        elif cache is None:
            row = runner.serve_uncached(
                request,
                storage=args.storage,
                block_tokens=args.block_tokens,
                cache_strategy=args.cache_strategy,
            )
        else:
            row = runner.serve(
                request,
                cache,
                storage=args.storage,
                block_tokens=args.block_tokens,
                cache_strategy=args.cache_strategy,
                validate_agreement=args.validate_agreement,
                agreement_atol=args.agreement_atol,
            )
        row.update({
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "request_index": index,
            "phase": "cold" if index < cold_requests else "steady",
            "cold_requests": cold_requests,
            "workload": args.workload,
            "seed": args.seed,
            "storage": args.storage,
            "device": args.device,
            "cache_device": (
                runner.cache_device(args.storage) if cache is not None else "none"
            ),
            "model": args.model,
            "execution_mode": "no-inference" if args.no_inference else "inference",
            "cache_strategy": args.cache_strategy if cache is not None else "none",
            "baseline_mode": args.baseline_mode if cache is None else None,
            "inference_path": (
                "article-cache"
                if cache is not None
                else f"{args.baseline_mode}-uncached"
            ),
            "l0_reused": cache is not None or args.baseline_mode == "segmented",
            "block_tokens": args.block_tokens,
            "prefill_is_simulated": args.no_inference,
            "offline_prefill_s": offline_prefill_s,
            "prefill_cost_model": runner.prefill_cost_model,
        })
        if reference_rows is not None:
            attach_offline_reference(
                row,
                reference_rows,
                storage=args.storage,
                agreement_atol=args.resolved_agreement_atol,
                strict=args.strict_reference,
            )
        rows.append(row)
        completed = index + 1
        if args.progress_every and (
            completed % args.progress_every == 0 or completed == len(trace)
        ):
            elapsed = time.perf_counter() - serving_started
            rate = completed / max(elapsed, 1e-12)
            print(
                f"[{args.output.name}] requests {completed}/{len(trace)} "
                f"({rate:.1f} req/s, occupancy={row.get('occupancy', 0.0):.1%}, "
                f"evictions={row.get('evictions', 0)})",
                flush=True,
            )
    write_jsonl(args.output, rows)
    summary = summarize(rows, cold_requests=cold_requests)
    summary.update({
        "offline_prefill_s": offline_prefill_s,
        "offline_amortized_prefill_s": offline_prefill_s / max(1, len(rows)),
        "split": args.split,
        "workload": args.workload,
        "seed": args.seed,
        "policy": args.policy,
        "storage": args.storage,
        "device": args.device,
        "cache_device": (
            runner.cache_device(args.storage) if cache is not None else "none"
        ),
        "model": args.model,
        "execution_mode": "no-inference" if args.no_inference else "inference",
        "cache_strategy": args.cache_strategy if cache is not None else "none",
        "baseline_mode": args.baseline_mode if cache is None else None,
        "inference_path": (
            "article-cache"
            if cache is not None
            else f"{args.baseline_mode}-uncached"
        ),
        "l0_reused": cache is not None or args.baseline_mode == "segmented",
        "block_tokens": args.block_tokens,
        "prefill_is_simulated": args.no_inference,
        "budget_bytes": budget_bytes,
        "budget_mb": args.budget_mb,
        "budget_percent": args.budget_percent,
        "working_set_bytes": corpus_working_set,
        "prefill_cost_model": runner.prefill_cost_model,
        "agreement_atol": args.resolved_agreement_atol,
        "reference_mode": (
            "offline-jsonl" if args.reference_jsonl is not None else None
        ),
        "strict_reference": args.strict_reference,
    })
    summary_json = args.output.with_suffix(".summary.json")
    summary_csv = args.output.with_suffix(".summary.csv")
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(summary_csv, [summary])
    manifest = _manifest(
        args, "no-inference" if args.no_inference else "inference", runner
    )
    write_manifest(args.output, manifest)
    write_manifest(summary_json, manifest)
    write_manifest(summary_csv, manifest)
    print(json.dumps(summary, indent=2))
    return 0


def _manifest(args, run_type, runner=None):
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "run_type": run_type,
        "dataset": str(args.dataset),
        "dataset_checksum": dataset_checksum(args.dataset),
        "split": args.split,
        "seed": getattr(args, "seed", getattr(args, "seeds", None)),
        "model": getattr(args, "model", None),
        "model_revision": (
            getattr(runner, "model_revision", None)
            or getattr(args, "model_revision", None)
            or getattr(args, "model", None)
        ),
        "tokenizer_revision": getattr(
            runner, "tokenizer_revision", getattr(args, "resolved_tokenizer_revision", None)
        ),
        "prompt_version": PROMPT_VERSION,
        "policy": getattr(args, "policy", getattr(args, "policies", None)),
        "workload": getattr(args, "workload", getattr(args, "workloads", None)),
        "storage_format": getattr(args, "storage", None),
        "device": getattr(args, "device", None),
        "cache_device": (
            runner.cache_device(args.storage)
            if (
                runner is not None
                and getattr(args, "storage", None) is not None
                and getattr(args, "policy", None) != "none"
            )
            else None
        ),
        "execution_mode": (
            "no-inference" if getattr(args, "no_inference", False) else run_type
        ),
        "cache_strategy": getattr(args, "cache_strategy", "document"),
        "baseline_mode": getattr(args, "baseline_mode", None),
        "block_tokens": getattr(args, "block_tokens", None),
        "dtype": getattr(args, "dtype", "float16"),
        "agreement_atol": getattr(args, "resolved_agreement_atol", None),
        "quantization_format": (
            "symmetric-int8-per-layer-per-kv-head"
            if getattr(args, "storage", None) == "cpu-int8" else "none"
        ),
        "budget_bytes": (
            getattr(args, "resolved_budget_bytes", None)
            if hasattr(args, "budget_mb")
            else None
        ),
        "budget_percent": getattr(args, "budget_percent", None),
        "budget_mb": getattr(args, "budget_mb", None),
        "prefill_calibration": (
            str(args.prefill_calibration)
            if getattr(args, "prefill_calibration", None) is not None
            else None
        ),
        "prefill_cost_model": getattr(runner, "prefill_cost_model", None),
        "reference_jsonl": (
            str(args.reference_jsonl)
            if getattr(args, "reference_jsonl", None) is not None
            else None
        ),
        "reference_checksum": (
            dataset_checksum(args.reference_jsonl)
            if getattr(args, "reference_jsonl", None) is not None
            else None
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
