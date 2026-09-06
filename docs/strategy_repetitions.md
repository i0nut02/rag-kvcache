# Bounded timing repetitions after simulation-based selection

Status: prepared, not measured. This is an optional repeatability check, not a
replacement for the frozen full-dev results or another broad search.

## Why these runs?

The project first ran the **108-combination no-inference test matrix**, then
used its capacity/reuse/policy results to choose **selected dev inference**.
Fixed-block sensitivity and selected inference established 256 tokens as the
tuned generic baseline. The full-dev results subsequently exposed small timing
differences between document, fixed-block and radix that a single execution
cannot establish as repeatable.

That selection remains intact. This notebook only repeats the organization
comparison. LRU is held constant to isolate organization; this is not a claim
that LRU is the best policy on every workload. GDSF, LFU, other budgets, grouped
locality, CPU quantization, arena and Triton are not swept again. See
[simulation findings](no_inference_results.md), the
[selected-inference protocol](inference_confirmation.md), and
[consolidated results](results.md).

Simulation measures reuse and management behavior, not observed GPU TTFT.
The new repetitions therefore use actual model inference.

## Protocol

Open [the notebook](../notebooks/strategy_repetitions_colab.ipynb) in Colab with
a **T4 runtime**. Its configuration is
[`strategy_repetitions.json`](../configs/strategy_repetitions.json).

| Parameter | Setting |
|---|---|
| Model | Qwen2.5-1.5B-Instruct, pinned revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Dataset | Official HTML-stripped QuALITY v1.0.1 dev |
| Queries/run | First 1,000 requests of each deterministic trace |
| Workloads | Random and Zipf(1.1), seed 42 |
| Caches | Document, fixed-block-256, radix |
| Policy / storage / capacity | LRU / GPU FP16 tensor backend / 4 GiB |
| Repetitions | Three per cache/workload, rotating cache execution order |
| References | One fresh segmented uncached reference per workload, for correctness |
| Planned cost | 18 cached runs + 2 references = 20,000 requests; roughly 7–9 hours |
| Functional smoke | 8 paths × 10 requests; separate artifacts, not timing evidence |

Each run has a fresh Python process and empty article cache. The model is not
trained. References are loaded from JSONL, avoiding extra forwards and the
associated cache-plus-reference CUDA memory spike. There is no per-repetition
uncached control: the new repeated timing claim compares cached strategies
directly, not repeated cache/control speedups.

Groups 1–2 each contain one reference and three caches. Groups 3–6 each contain
three caches, reusing the corresponding correctness reference. Cache order
rotates document/fixed/radix, fixed/radix/document, radix/document/fixed.

The estimate scales earlier measured latencies; a slow runtime may not finish
everything within the cap. The previous 19-hour proposal repeated 2,086
questions across 24 configurations. This protocol deliberately reduces both
query count and redundant reference computation.

## What the ten-hour limit means

The notebook's deadline starts with its first smoke/benchmark process. It
includes model loading, first-time model downloads, checkpointing, pauses and
retries after that start. Installation and dataset download before the smoke
are outside this benchmark window. The deadline is saved as
`results/strategy_repetitions/time-budget.json` and **does not reset** when a
cell is re-run or its checkpoint restored.

At the deadline, a watchdog kills the active benchmark process group even if
it is no longer printing. Further inference is refused. An interrupted run is
not a finished measurement. This stops benchmark processes, **not Colab
billing**: disconnect/delete the runtime when finished.

To keep this bound, use the notebook execution cells, not a bare
`matrix --execute`: the ordinary matrix command does not enforce this
notebook's deadline or completion receipts. There is intentionally no `full`
profile in the new config. `confirmation` means 1,000 queries here, and `smoke`
means ten; older configs retain their original profile meanings.

## Checkpoints and validation

No Google Drive mount is needed. A local ZIP is refreshed after each completed
run and downloaded through the browser after each group. Download and verify
it before ending the runtime. It includes raw outputs, manifests, environment
and protocol snapshots, completion receipts and the deadline, not model
weights or dataset files.

Resume checks the command, artifact hashes, model/tokenizer revision, source
snapshot, packages and GPU/driver identity. Use the saved code revision when
restoring; do not pull a new implementation in the middle of the study.
Completed runs are skipped; an incomplete run restarts from request zero.
Restoring after the deadline permits analysis, but no further inference.

The notebook is intentionally separate from the frozen full-dev archive.
Current radix access accounting and result schema differ from the historical
measurements. Those results remain valid evidence for their recorded revision,
but must not be imported as repetition 1 of this study.

## Analysis and report use

The notebook writes `run_metrics.csv`, `strategy_variation.csv`,
`paired_runs.csv`, `paired_variation.csv`, input hashes, coverage and plots.
It calculates **mean and p90 TTFT for each run**, then descriptive medians,
ranges and sample standard deviations across repetitions. Paired reductions
compare caches within the same workload/repetition; positive reduction means
the candidate is faster than the baseline cache. The single references are
shown once, not replicated to create artificial sample size.

All-request timing is primary. A separate diagnostic drops the first 100
requests without resetting state. Do not call this exclusion a guarantee that
the cache is stationary, and do not replace the primary result with whichever
scope favors a hypothesis.

If the deadline prevents completion, set `ALLOW_INCOMPLETE = True` in the
analysis cell. It only compares **complete three-cache groups** and reports
actual repetition counts; no half-finished trace or unmatched strategy enters
the comparison. Missing and excluded runs remain visible in provenance.

The 1,000-request subset is not full-dev accuracy, and Zipf repeats Q&As.
Shared cache state also makes requests dependent, so there is **no IID
request-level bootstrap**. Even three consistent rankings are limited timing
evidence, not a significance claim or multi-seed/hardware generalization.
Keep small differences qualified, particularly if the winner changes between
repetitions. Preserve the full-dev evidence as the main accuracy evaluation.
