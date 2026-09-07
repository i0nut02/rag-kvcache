# No-inference matrix results

## Scope and provenance

This document records the completed label-free cache trace experiment. The
source artifact is `results/test_matrix/full/all_summaries.csv`; generated
results remain outside Git and must be archived with the final report.

Historical implementation caveat: these radix rows predate the 5 September
2026 access-accounting fix.
Insertion used to perform an extra policy-mutating lookup. Preserve these rows
with their original revision; do not combine them with corrected-radix runs or
use them as measured overhead/rankings of the corrected implementation.

- Code revision: `07bc635` (`Optimize fixed-block cache eviction`)
- Dataset: QuALITY v1.0.1 HTML-stripped test split
- Dataset SHA-256: `ca103a953741c56888124a14958460b07941ee40a844914d39e851f1c3099897`
- Requests per run: 2,128 across 116 articles
- Model geometry/tokenizer: `Qwen/Qwen2.5-1.5B-Instruct`
- Workloads: grouped, random seed 42, and Zipf(1.1) seed 42
- Cache strategies: document, 16-token fixed-block, and radix
- Policies: LRU, LFU, and GDSF
- Storage accounting: accelerator FP16 and CPU INT8
- Tensor-byte budgets: 4 GiB and 8 GiB
- Total combinations: 108

`--no-inference` loads the tokenizer and model configuration but never loads
model weights, allocates KV tensors, or calls a model forward pass. Accelerator
and INT8 capacity are therefore exact geometry-based accounting simulations.
Prefill cost is estimated by the measured CUDA piecewise calibration stored in
`configs/prefill_qwen2.5_1.5b_cuda.json`.

## Best article-token hit rate

The table reports the best strategy/policy within every workload, storage, and
budget group. Values are percentages of requested article tokens restored from
cache; the question and answer options are never cacheable.

| Workload | FP16 4 GiB | FP16 8 GiB | INT8 4 GiB | INT8 8 GiB |
|---|---:|---:|---:|---:|
| Grouped | 94.55% | 94.55% | 94.55% | 94.55% |
| Random | 23.44% | 43.78% | 43.71% | 84.89% |
| Zipf | 70.66% | 84.35% | 84.16% | 94.52% |

The selected configurations were:

- Grouped: all organizations and policies were effectively tied. The maximum
  rows happened to be radix/GDSF because of negligible shared-prefix tokens.
- Random: fixed-block/LRU at every capacity.
- Zipf: document/GDSF at FP16 4 GiB and INT8 4 GiB, fixed-block/GDSF at
  FP16 8 GiB, and document/LRU at INT8 8 GiB after the cache approached
  saturation.

## Main findings

### INT8 approximately doubles useful capacity

Matched effective-capacity comparisons are almost identical:

| Workload | FP16 8 GiB | INT8 4 GiB | Absolute difference |
|---|---:|---:|---:|
| Random | 43.776% | 43.710% | 0.066 percentage points |
| Zipf | 84.353% | 84.164% | 0.189 percentage points |

The peak cached-token counts also scale as expected: about 149.8k tokens for
FP16 4 GiB, 299.6k for FP16 8 GiB or INT8 4 GiB, and 598.0k for INT8 8 GiB.
This validates byte and scale accounting. It does not yet establish INT8
latency or accuracy; those require actual restoration and inference.

### Workload locality dominates

Grouped order reaches the dataset reuse ceiling because all questions for an
article are consecutive. Random order continuously churns the cache and is the
most capacity-sensitive trace. Zipf benefits from its stable hot set and sits
between the two.

### Policy depends on the workload

LRU is consistently best for the random trace. GDSF is generally best for a
memory-constrained Zipf trace, with LFU sometimes close. Once effective
capacity approaches the working set, policy differences disappear. GDSF is
therefore useful for skewed reuse but is not a universal replacement for LRU.

### Document caching is a strong systems tradeoff

The three organizations differ much less than storage capacity or workload.
For random FP16 at 4 GiB with LRU:

| Strategy | Article-token hit | Lookup/request | Policy/request | Peak metadata |
|---|---:|---:|---:|---:|
| Document | 23.172% | 0.004 ms | 0.021 ms | 20 KiB |
| Radix | 23.233% | 0.128 ms | 0.527 ms | 5.2 MiB |
| Fixed-block | 23.444% | 0.653 ms | 6.024 ms | 9.8 MiB |

Fixed-block gains only 0.27 percentage points over atomic document caching in
this case. Radix shared just 516,096 bytes at peak, roughly 0.012% of a 4 GiB
budget, because unrelated QuALITY articles have almost no reusable text prefix.
The measurements above are Python policy-simulator overhead, not CUDA kernel
latency, but they demonstrate the metadata and management complexity clearly.

The trace result therefore supports the project hypothesis: when document
identity and boundaries are known, atomic document caching can retain nearly
all useful reuse with dramatically simpler metadata and policy management.

## Metric interpretation

- Primary trace metrics are `request_hit_rate`, `article_token_hit_rate`, and
  `byte_hit_rate`.
- `request_hit_rate` includes the pinned L0 system prefix in the numerator and
  the uncached question/options suffix in the denominator.
- `article_token_hit_rate` excludes L0 and is the cleanest cross-strategy reuse
  comparison.
- `partial_prefix_hit_rate` is always one because L0 is pinned; it must not be
  presented as evidence of article reuse.
- `partial_article_hit_rate` is a compatibility alias for a partial L0-to-
  document-tree hit. Use `partial_article_text_hit_rate` for actual article
  prefix reuse.
- Document caching has zero partial article-text hits by design because its
  eviction unit is atomic.
- Fixed-block stores complete blocks only, so `full_document_hit_rate` is not a
  fair cross-strategy metric when an article has a trailing partial block.
- `cache_bytes_peak` obeys the tensor budget. `cache_footprint_bytes_peak` also
  includes Python metadata and may exceed that tensor budget slightly.
- `prefill_time_s` and `amortized_prefill_time_s` are calibrated simulated
  costs, not observed wall time or TTFT.
- `accuracy`, `quality_hard_accuracy`, cached-label agreement, TTFT, transfer,
  dequantization, and CUDA/MPS allocation are unavailable in this experiment
  and correctly appear as `NaN` or zero.

## Completed inference follow-up

The labelled QuALITY dev confirmation, fixed-block sensitivity, 300-request
INT8 comparison, timing repetitions, and complete 2,086-request inference
suite are finished. The complete suite is now primary; the smaller experiments
remain supporting evidence. Its main conclusions are:

- document/LRU FP16 gives `1.281x` random and `2.741x` Zipf mean speedup over
  all 2,086 requests;
- 256 tokens is the selected fixed-block size; the historical full-dev row
  favors document/LRU on random and radix on Zipf, while the later three-run
  check finds similar document/radix latency and consistently slower fixed-256;
- CPU INT8 gives `1.578x` random speedup and changes 12/2,086 labels, with no
  net overall-accuracy change but a two-answer hard-accuracy loss.

See [`results.md`](results.md) for the consolidated tables, metric caveats,
mismatch details, timing ranges, and follow-up archive hashes. The original
selected-suite procedure remains in
[`inference_confirmation.md`](inference_confirmation.md).
The completed 1,000-request repetitions, selected through this earlier
simulation/inference sequence, are summarized in [`results.md`](results.md).
