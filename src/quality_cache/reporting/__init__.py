"""Metrics, result serialization, manifests, and plots."""

from .io import write_csv, write_jsonl
from .inference_analysis import analyze_inference_confirmation
from .manifest import write_manifest
from .metrics import group_summaries, label_agreement, summarize
from .plots import make_primary_figures

__all__ = [
    "group_summaries",
    "analyze_inference_confirmation",
    "label_agreement",
    "make_primary_figures",
    "summarize",
    "write_csv",
    "write_jsonl",
    "write_manifest",
]
