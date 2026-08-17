"""Expansion and execution of the configured QuALITY test-only matrix."""

from __future__ import annotations

import itertools
import json
import shlex
from pathlib import Path
from typing import Callable


def add_matrix_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "matrix", help="expand or execute the configured test-only cache matrix"
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
    required = (
        "dataset",
        "split",
        "model",
        "tokenizer",
        "cache_strategies",
        "policies",
        "storage",
        "workloads",
        "seeds",
        "budget_mb",
        "profiles",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"matrix config is missing required fields: {missing}")
    if payload["split"] != "test":
        raise ValueError("this matrix must use only the QuALITY test split")
    if not payload.get("no_inference", False):
        raise ValueError("QuALITY test labels are withheld; matrix requires no_inference=true")
    if not Path(payload["dataset"]).is_file():
        raise ValueError(f"test dataset does not exist: {payload['dataset']}")
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


def run_matrix(args, run_command: Callable[[list[str]], int]) -> int:
    config = load_matrix(args.config)
    output_dir = args.output_dir or Path("results/test_matrix") / args.profile
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
                "no_inference": True,
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
        result = run_command(command)
        if result:
            return result
    return 0
