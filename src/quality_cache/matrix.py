"""Expansion and execution of grid and selected QuALITY experiment matrices."""

from __future__ import annotations

import gc
import itertools
import json
import re
import shlex
from pathlib import Path
from typing import Callable


def add_matrix_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "matrix", help="expand or execute a configured cache experiment matrix"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--profile", choices=["smoke", "confirmation", "full"], default="smoke"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--execute", action="store_true", help="run commands instead of only validating the matrix"
    )
    parser.add_argument(
        "--resume", action="store_true", help="skip combinations whose JSONL output already exists"
    )
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--show-commands", action="store_true")


def load_matrix(path: str | Path) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "dataset",
        "split",
        "model",
        "profiles",
    ]
    selected = "runs" in payload
    if selected:
        required.append("runs")
    else:
        required.extend(
            (
                "tokenizer",
                "cache_strategies",
                "policies",
                "storage",
                "workloads",
                "seeds",
                "budget_mb",
            )
        )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"matrix config is missing required fields: {missing}")
    if not Path(payload["dataset"]).is_file():
        raise ValueError(f"matrix dataset does not exist: {payload['dataset']}")
    if selected:
        _validate_selected_runs(payload)
    else:
        if payload["split"] != "test":
            raise ValueError("the full cache grid must use only the QuALITY test split")
        if not payload.get("no_inference", False):
            raise ValueError(
                "QuALITY test labels are withheld; the full grid requires no_inference=true"
            )
    return payload


def build_matrix_commands(
    config: dict,
    profile: str,
    output_dir: str | Path,
) -> list[list[str]]:
    try:
        profile_config = config["profiles"][profile]
    except KeyError as error:
        raise ValueError(f"matrix profile is not configured: {profile}") from error
    output_dir = Path(output_dir)
    if "runs" in config:
        return _build_selected_commands(config, profile, profile_config, output_dir)
    commands: list[list[str]] = []
    combinations = itertools.product(
        config["cache_strategies"],
        config["policies"],
        config["storage"],
        config["workloads"],
        config["seeds"],
        config["budget_mb"],
    )
    block_tokens = config.get("block_tokens", {})
    for strategy, policy, storage, workload, seed, budget_mb in combinations:
        budget_label = f"{int(budget_mb) // 1024}gib"
        storage_label = storage.replace("-", "_")
        output = output_dir / (
            f"test_{profile}_{strategy}_{policy}_{storage_label}_"
            f"{workload}_seed{seed}_{budget_label}.jsonl"
        )
        command = [
            "run",
            str(config["dataset"]),
            "--split",
            "test",
            "--verify-counts",
            "--model",
            str(config["model"]),
            "--tokenizer",
            str(config["tokenizer"]),
            "--device",
            str(config.get("device", "mps")),
            "--dtype",
            str(config.get("dtype", "float16")),
            "--cache-strategy",
            str(strategy),
            "--policy",
            str(policy),
            "--storage",
            str(storage),
            "--budget-mb",
            str(budget_mb),
            "--workload",
            str(workload),
            "--seed",
            str(seed),
            "--block-tokens",
            str(block_tokens.get(strategy, 16)),
            "--no-inference",
            "--output",
            str(output),
        ]
        limit = profile_config.get("limit")
        if limit is not None:
            command.extend(("--limit", str(limit)))
        calibration = config.get("prefill_calibration")
        if calibration:
            command.extend(("--prefill-calibration", str(calibration)))
        commands.append(command)
    return commands


def _validate_selected_runs(config: dict) -> None:
    runs = config["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("selected matrix requires a non-empty runs list")
    required = ("name", "workload", "policy", "storage", "cache_strategy")
    names = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"selected run {index} must be an object")
        missing = [field for field in required if field not in run]
        if missing:
            raise ValueError(f"selected run {index} is missing fields: {missing}")
        name = str(run["name"])
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None:
            raise ValueError(f"selected run has unsafe name: {name!r}")
        names.append(name)
        no_inference = bool(
            run.get("no_inference", config.get("no_inference", False))
        )
        if config["split"] == "test" and not no_inference:
            raise ValueError(
                f"selected run {name!r} requires labelled train or dev data"
            )
        if run["policy"] != "none" and run.get(
            "budget_mb", config.get("budget_mb")
        ) is None:
            raise ValueError(f"selected cached run {name!r} requires budget_mb")
        if (
            run.get("validate_agreement", config.get("validate_agreement", False))
            and no_inference
        ):
            raise ValueError(
                f"selected run {name!r} cannot validate agreement without inference"
            )
    if len(names) != len(set(names)):
        raise ValueError("selected run names must be unique")
    positions = {name: index for index, name in enumerate(names)}
    for index, run in enumerate(runs):
        reference_name = run.get("reference_run")
        if reference_name is None:
            continue
        name = str(run["name"])
        if run.get("validate_agreement", config.get("validate_agreement", False)):
            raise ValueError(
                f"selected run {name!r} cannot use both reference_run and "
                "validate_agreement"
            )
        if reference_name not in positions:
            raise ValueError(
                f"selected run {name!r} has unknown reference_run {reference_name!r}"
            )
        if positions[reference_name] >= index:
            raise ValueError(
                f"reference_run {reference_name!r} must precede selected run {name!r}"
            )
        reference = runs[positions[reference_name]]
        if reference["policy"] != "none":
            raise ValueError(f"reference_run {reference_name!r} must be uncached")
        if reference["workload"] != run["workload"]:
            raise ValueError(
                f"selected run {name!r} and its reference must share a workload"
            )
        if reference.get("seed", config.get("seed", 42)) != run.get(
            "seed", config.get("seed", 42)
        ):
            raise ValueError(
                f"selected run {name!r} and its reference must share a seed"
            )


def _build_selected_commands(
    config: dict,
    profile: str,
    profile_config: dict,
    output_dir: Path,
) -> list[list[str]]:
    commands = []
    for run in config["runs"]:
        name = str(run["name"])
        strategy = str(run["cache_strategy"])
        policy = str(run["policy"])
        storage = str(run["storage"])
        no_inference = bool(
            run.get("no_inference", config.get("no_inference", False))
        )
        block_setting = run.get("block_tokens", config.get("block_tokens", 16))
        if isinstance(block_setting, dict):
            block_setting = block_setting.get(strategy, 16)
        output = output_dir / f"{config['split']}_{profile}_{name}.jsonl"
        command = [
            "run",
            str(config["dataset"]),
            "--split",
            str(config["split"]),
        ]
        if config.get("verify_counts", False):
            command.append("--verify-counts")
        command.extend(("--model", str(config["model"])))
        if config.get("model_revision"):
            command.extend(("--model-revision", str(config["model_revision"])))
        if config.get("tokenizer"):
            command.extend(("--tokenizer", str(config["tokenizer"])))
        command.extend(
            (
                "--device",
                str(config.get("device", "mps")),
                "--dtype",
                str(config.get("dtype", "float16")),
                "--cache-strategy",
                strategy,
                "--policy",
                policy,
                "--storage",
                storage,
                "--workload",
                str(run["workload"]),
                "--seed",
                str(run.get("seed", config.get("seed", 42))),
                "--block-tokens",
                str(block_setting),
            )
        )
        if policy != "none":
            command.extend(
                ("--budget-mb", str(run.get("budget_mb", config.get("budget_mb"))))
            )
        if run.get("max_articles") is not None:
            command.extend(("--max-articles", str(run["max_articles"])))
        if run.get("offline_prefill", config.get("offline_prefill", False)):
            command.append("--offline-prefill")
        if no_inference:
            command.append("--no-inference")
        if run.get("validate_agreement", config.get("validate_agreement", False)):
            command.append("--validate-agreement")
        reference_name = run.get("reference_run")
        if reference_name is not None:
            reference_output = (
                output_dir
                / f"{config['split']}_{profile}_{reference_name}.jsonl"
            )
            command.extend(("--reference-jsonl", str(reference_output)))
        agreement_atol = run.get("agreement_atol", config.get("agreement_atol"))
        if agreement_atol is not None:
            command.extend(("--agreement-atol", str(agreement_atol)))
        progress_every = run.get("progress_every", config.get("progress_every"))
        if progress_every is not None:
            command.extend(("--progress-every", str(progress_every)))
        limit = profile_config.get("limit")
        if limit is not None:
            command.extend(("--limit", str(limit)))
        calibration = run.get(
            "prefill_calibration", config.get("prefill_calibration")
        )
        if calibration and no_inference:
            command.extend(("--prefill-calibration", str(calibration)))
        command.extend(("--output", str(output)))
        commands.append(command)
    return commands


def run_matrix(args, run_command: Callable[[list[str]], int]) -> int:
    config = load_matrix(args.config)
    output_root = Path(config.get("output_root", "results/test_matrix"))
    output_dir = args.output_dir or output_root / args.profile
    commands = build_matrix_commands(config, args.profile, output_dir)
    if args.max_runs is not None:
        if args.max_runs <= 0:
            raise ValueError("--max-runs must be positive")
        commands = commands[: args.max_runs]
    print(
        json.dumps(
            {
                "dataset": config["dataset"],
                "split": config["split"],
                "profile": args.profile,
                "no_inference": bool(config.get("no_inference", False)),
                "matrix_type": "selected" if "runs" in config else "grid",
                "combinations": len(commands),
                "output_dir": str(output_dir),
                "execute": args.execute,
            },
            indent=2,
        )
    )
    if args.show_commands:
        for command in commands:
            print("python experiments/run_quality.py " + shlex.join(command))
    if not args.execute:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, command in enumerate(commands, start=1):
        output = Path(command[command.index("--output") + 1])
        if args.resume and output.exists():
            print(f"[{index}/{len(commands)}] skip {output}")
            continue
        print(f"[{index}/{len(commands)}] run {output.name}")
        try:
            result = run_command(command)
        finally:
            if "--no-inference" not in command:
                _release_accelerator_memory()
        if result:
            return result
    return 0


def _release_accelerator_memory() -> None:
    """Release tensors and allocator reservations between in-process runs."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if hasattr(torch, "mps") and mps_backend and mps_backend.is_available():
        torch.mps.empty_cache()
