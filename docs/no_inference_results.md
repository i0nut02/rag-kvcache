# No-inference matrix results

## Scope and provenance

This document records the completed label-free cache trace experiment. The
source artifact is `results/test_matrix/full/all_summaries.csv`; generated
results remain outside Git and must be archived with the final report.

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

## Remaining confirmation work

The next phase uses the labelled QuALITY dev split and real CUDA inference on a
small selected suite. It must measure uncached and cached TTFT, FP16 numerical
agreement, CPU INT8 restoration/dequantization cost, label agreement, overall
accuracy, and QuALITY-hard accuracy. A separate no-inference sensitivity check
should compare fixed-block sizes 16, 64, and 256 without repeating the entire
108-run matrix. The selected next phase is encoded in
[`../configs/inference_confirmation.json`](../configs/inference_confirmation.json)
and documented in [`inference_confirmation.md`](inference_confirmation.md).
