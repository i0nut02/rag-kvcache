# Fair inference confirmation

Analysis schema: `quality-fair-inference-v1`. The archive contains 10 aligned runs using `Qwen/Qwen2.5-1.5B-Instruct` and Torch `2.11.0+cu128`.

The cache-only baseline is segmented inference with pinned L0 and no retained article KV. End-to-end speedup uses the original one-forward uncached path.

## Latency

| Workload | Strategy | Storage | Cached TTFT mean/p50/p95 (s) | Cache-only speedup (95% CI) | End-to-end speedup | Cache-only TTFT change |
|---|---|---|---:|---:|---:|---:|
| random | document / lru | accelerator-fp16 | 1.634 / 2.063 / 2.696 | 1.19x [1.08, 1.33] | 3.72x | +15.9% |
| random | fixed-block / lru | accelerator-fp16 | 2.335 / 2.764 / 3.726 | 0.83x [0.77, 0.91] | 2.60x | -20.2% |
| random | radix / lru | accelerator-fp16 | 1.635 / 2.089 / 2.702 | 1.19x [1.08, 1.33] | 3.71x | +15.8% |
| random | document / lru | cpu-int8 | 1.456 / 1.694 / 2.783 | 1.33x [1.18, 1.54] | 4.17x | +25.0% |
| zipf | document / lru | accelerator-fp16 | 0.957 / 0.064 / 2.523 | 2.19x [1.81, 2.78] | 6.52x | +54.4% |
| zipf | document / gdsf | accelerator-fp16 | 0.959 / 0.065 / 2.525 | 2.19x [1.81, 2.77] | 6.51x | +54.3% |

A positive TTFT change is a reduction. A cache-only speedup below 1.0 means cache management made the run slower than segmented execution without document retention.

## Correctness against the segmented path

| Workload | Strategy | Storage | Label agreement | Accuracy (segmented -> cache) | Hard accuracy (segmented -> cache) | Mismatches | Maximum logit delta |
|---|---|---|---:|---:|---:|---:|---:|
| random | Document LRU FP16 | accelerator-fp16 | 100.0% | 51.0% -> 51.0% (+0.0 pp) | 35.7% -> 35.7% (+0.0 pp) | 0 | 0.000000 |
| random | Fixed-block LRU FP16 | accelerator-fp16 | 100.0% | 51.0% -> 51.0% (+0.0 pp) | 35.7% -> 35.7% (+0.0 pp) | 0 | 0.046875 |
| random | Radix LRU FP16 | accelerator-fp16 | 100.0% | 51.0% -> 51.0% (+0.0 pp) | 35.7% -> 35.7% (+0.0 pp) | 0 | 0.062500 |
| random | Document LRU CPU INT8 | cpu-int8 | 99.0% | 51.0% -> 50.0% (-1.0 pp) | 35.7% -> 33.9% (-1.8 pp) | 1 | 0.828125 |
| zipf | Document LRU FP16 | accelerator-fp16 | 100.0% | 60.0% -> 60.0% (+0.0 pp) | 51.8% -> 51.8% (+0.0 pp) | 0 | 0.000000 |
| zipf | Document GDSF FP16 | accelerator-fp16 | 100.0% | 60.0% -> 60.0% (+0.0 pp) | 51.8% -> 51.8% (+0.0 pp) | 0 | 0.000000 |

The FP16 document-cache paths match segmented execution exactly. CPU INT8 has 1 label mismatch in this 100-request sample, so its quality effect needs a larger confirmation before it is described as lossless.

## Segmented control against the full path

| Workload | Label agreement | Accuracy (full -> segmented) | Hard accuracy (full -> segmented) | Mismatch occurrences | Unique questions | Maximum logit delta |
|---|---:|---:|---:|---:|---:|---:|
| random | 100.0% | 51.0% -> 51.0% (+0.0 pp) | 35.7% -> 35.7% (+0.0 pp) | 0 | 0 | 0.125000 |
| zipf | 98.0% | 62.0% -> 60.0% (-2.0 pp) | 55.4% -> 51.8% (-3.6 pp) | 2 | 1 | 0.093750 |

The 2 Zipf mismatch occurrences represent 1 unique question. Both FP16 document-cache runs match segmented execution exactly, so this discrepancy comes from the segmented forward path rather than cache restoration or eviction.

## Figures

- [TTFT by execution path](ttft_by_execution_path.pdf)
- [Cache-only speedup](cache_only_speedup.pdf)
- [Article-hit/latency tradeoff](hit_latency_tradeoff.pdf)
