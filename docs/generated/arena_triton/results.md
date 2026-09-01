# Fair inference confirmation

Analysis schema: `quality-fair-inference-v2`. The archive contains 6 aligned runs using `Qwen/Qwen2.5-1.5B-Instruct` and Torch `2.11.0+cu128`. Result schema: `quality-kv-v3`.

The cache-only baseline is segmented inference with pinned L0 and no retained article KV. End-to-end values are reported only when the suite also includes a full one-forward control.

## Latency

| Workload | Path | Storage | Cached TTFT mean/p50/p95 (s) | Cache-only speedup (95% CI) | End-to-end speedup | Cache-only TTFT change |
|---|---|---|---:|---:|---:|---:|
| random | Document tensor FP16 | accelerator-fp16 | 1.544 / 1.944 / 2.532 | 1.26x [1.14, 1.41] | n/a | +20.4% |
| random | Document arena-64 FP16 | accelerator-fp16 | 1.723 / 2.164 / 2.936 | 1.13x [1.02, 1.26] | n/a | +11.2% |
| random | Document arena-256 FP16 | accelerator-fp16 | 1.687 / 2.097 / 2.759 | 1.15x [1.05, 1.28] | n/a | +13.1% |
| random | Document CPU INT8, PyTorch restore | cpu-int8 | 1.438 / 1.708 / 2.848 | 1.35x [1.19, 1.57] | n/a | +25.9% |
| random | Document CPU INT8, Triton restore | cpu-int8 | 1.418 / 1.671 / 2.787 | 1.37x [1.20, 1.59] | n/a | +26.9% |

A positive TTFT change is a reduction. A cache-only speedup below 1.0 means cache management made the run slower than segmented execution without document retention.

## Correctness against the segmented path

| Workload | Strategy | Storage | Label agreement | Accuracy (segmented -> cache) | Hard accuracy (segmented -> cache) | Mismatches | Maximum logit delta |
|---|---|---|---:|---:|---:|---:|---:|
| random | Document tensor FP16 | accelerator-fp16 | 100.0% | 51.0% -> 51.0% (+0.0 pp) | 35.7% -> 35.7% (+0.0 pp) | 0 | 0.000000 |
| random | Document arena-64 FP16 | accelerator-fp16 | 100.0% | 51.0% -> 51.0% (+0.0 pp) | 35.7% -> 35.7% (+0.0 pp) | 0 | 0.000000 |
| random | Document arena-256 FP16 | accelerator-fp16 | 100.0% | 51.0% -> 51.0% (+0.0 pp) | 35.7% -> 35.7% (+0.0 pp) | 0 | 0.000000 |
| random | Document CPU INT8, PyTorch restore | cpu-int8 | 99.0% | 51.0% -> 50.0% (-1.0 pp) | 35.7% -> 33.9% (-1.8 pp) | 1 | 0.828125 |
| random | Document CPU INT8, Triton restore | cpu-int8 | 99.0% | 51.0% -> 50.0% (-1.0 pp) | 35.7% -> 33.9% (-1.8 pp) | 1 | 0.828125 |

## Figures

- [TTFT by execution path](ttft_by_execution_path.pdf)
- [Cache-only speedup](cache_only_speedup.pdf)
- [Article-hit/latency tradeoff](hit_latency_tradeoff.pdf)
