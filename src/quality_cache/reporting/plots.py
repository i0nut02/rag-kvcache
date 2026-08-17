"""Primary experiment figure generation."""

from __future__ import annotations

import csv
from pathlib import Path


def make_primary_figures(summary_csv: str | Path, output_dir: str | Path) -> list[Path]:
    import matplotlib.pyplot as plt

    with Path(summary_csv).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    for row in rows:
        row["policy_workload"] = f"{row.get('policy', '')}/{row.get('workload', '')}"
        row["storage_policy"] = f"{row.get('storage', '')}/{row.get('policy', '')}"
        row["budget_gib"] = _number(row.get("budget_bytes")) / 2**30
        footprint = row.get("cache_footprint_bytes_peak") or row.get("cache_bytes_peak")
        row["cache_footprint_gib"] = _number(footprint) / 2**30

    figures.append(_line_plot(
        plt, rows, output_dir / "hit_rate_vs_budget.pdf", "budget_gib",
        "request_hit_rate", "policy_workload", "Hit rate versus cache budget", "Request hit rate"
    ))
    ttft_rows = [row for row in rows if _number(row.get("ttft_p50_s")) >= 0]
    figures.append(_line_plot(
        plt, ttft_rows, output_dir / "ttft_vs_budget.pdf", "budget_bytes",
        "ttft_p50_s", "storage_policy", "TTFT versus cache budget", "TTFT p50 (s)"
    ))
    four_gib = [row for row in rows if _number(row.get("budget_mb")) == 4096]
    figures.append(_grouped_bar(
        plt, four_gib or rows, output_dir / "policy_by_workload.pdf", "workload", "request_hit_rate",
        "policy", "Policy comparison by workload", "Request hit rate"
    ))
    tradeoff = [
        row
        for row in rows
        if row.get("storage") in {"cpu-fp16", "cpu-int8", "accelerator-fp16"}
    ]
    figures.append(_scatter(
        plt, tradeoff, output_dir / "storage_tradeoff.pdf",
        "cache_footprint_gib", "ttft_p50_s", "accuracy", "storage",
        "Memory–latency–accuracy tradeoff"
    ))
    return figures


def _line_plot(plt, rows, path, x, y, series, title, ylabel):
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for value in sorted({row.get(series, "") for row in rows}):
        grouped = {}
        for row in rows:
            if row.get(series, "") != value:
                continue
            grouped.setdefault(_number(row.get(x)), []).append(_number(row.get(y)))
        points = sorted(
            (x_value, sum(y_values) / len(y_values))
            for x_value, y_values in grouped.items()
        )
        if points:
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=value)
    axis.set(title=title, xlabel=x.replace("_", " ").title(), ylabel=ylabel)
    if rows:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _grouped_bar(plt, rows, path, x, y, series, title, ylabel):
    categories = sorted({row.get(x, "") for row in rows})
    groups = sorted({row.get(series, "") for row in rows})
    fig, axis = plt.subplots(figsize=(7.2, 4.0))
    width = 0.8 / max(1, len(groups))
    for index, group in enumerate(groups):
        values = []
        for category in categories:
            matches = [row for row in rows if row.get(x) == category and row.get(series) == group]
            values.append(sum(_number(row.get(y)) for row in matches) / max(1, len(matches)))
        positions = [i - 0.4 + width / 2 + index * width for i in range(len(categories))]
        axis.bar(positions, values, width, label=group)
    axis.set_xticks(range(len(categories)), categories, rotation=15)
    axis.set(title=title, ylabel=ylabel)
    if groups:
        axis.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _scatter(plt, rows, path, x, y, size, series, title):
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    for value in sorted({row.get(series, "") for row in rows}):
        selected = [row for row in rows if row.get(series) == value]
        axis.scatter(
            [_number(row.get(x)) for row in selected],
            [_number(row.get(y)) for row in selected],
            s=[30 + 100 * max(0, _number(row.get(size))) for row in selected],
            label=value,
        )
    axis.set(title=title, xlabel="Cache footprint (GiB)", ylabel="TTFT p50 (s)")
    if rows:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0
