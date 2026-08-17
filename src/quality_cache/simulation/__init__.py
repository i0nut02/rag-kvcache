"""Inference-free cache trace simulation."""

from .engine import (
    article_sizes,
    approximate_tokens,
    farthest_next_use_rows,
    simulate_trace,
    working_set_bytes,
)

__all__ = [
    "approximate_tokens",
    "article_sizes",
    "farthest_next_use_rows",
    "simulate_trace",
    "working_set_bytes",
]
