# Consolidated experimental results

Status: 18 August 2026. This is the report-ready interpretation of the
completed QuALITY trace, inference, block-size, INT8, and timing-repetition
experiments. Detailed source tables and figures for the first confirmation are
in [`generated/inference_confirmation`](generated/inference_confirmation/results.md).

## Executive result

The evidence supports document-aware KV caching for repeated long-document QA.
At a 4 GiB accelerator-FP16 budget, atomic document caching reduces mean TTFT
relative to the fair segmented no-document-cache control by about 19% on the
random trace and 54% on the Zipf trace. Across three timing repetitions, the
median paired speedups are `1.250x` and `2.194x`, respectively.

The generic alternatives do not improve this workload. Radix caching matches
document-cache latency and useful hit rate but has more metadata and lookup
work. A 16-token fixed-block cache is slower because of extreme block churn;
increasing the block to 256 tokens makes it competitive, but it remains 2.7%
slower than document caching on random traffic and 5.8% slower on Zipf traffic.
This is the main systems result: known document boundaries are useful semantic
information for choosing the cache allocation and eviction unit.

CPU INT8 approximately doubles capacity and, over 300 random requests, lowers
mean TTFT by 36.3% relative to segmented FP16. It agrees on 298/300 labels
(`99.33%`) but changes two answers that FP16 got right. It is therefore a
measured memory/latency/quality tradeoff, not a lossless mode.

## Research questions and answers

| Question | Answer from the current evidence |
|---|---|
| Does cross-request article-KV reuse reduce TTFT? | Yes: median paired speedup is `1.250x` on random and `2.194x` on Zipf traffic across three runs. |
| When do policies differ? | Workload locality and capacity dominate. LRU is strongest on the random trace; GDSF helps some constrained Zipf traces, but LRU and GDSF have indistinguishable 100-request FP16 TTFT. |
| Does INT8 increase useful capacity? | Yes. INT8 4 GiB closely matches FP16 8 GiB in the full trace simulation. |
| Is INT8 accuracy-neutral? | No strict lossless claim is justified: 2/300 labels changed and accuracy fell by 0.67 percentage points. |
| Does document-aware caching beat generic prefix structures? | Yes as a systems tradeoff: it gives the best measured TTFT with tiny metadata while retaining nearly the same useful tokens. |
| Is offline precomputation amortized? | Not yet answered experimentally. Calibrated prefill estimates are available, but no measured online-versus-offline prefill comparison should be claimed. |

## Experimental layers

The project uses two separate dataset roles and does not mix their metrics:

| Layer | Split | Requests | Purpose |
|---|---|---:|---|
| Complete no-inference matrix | Test | 108 runs x 2,128 | Cache capacity, hit rate, policy, and metadata behavior without labels or model forwards |
| Initial inference confirmation | Dev | 10 aligned runs x 100 | Fair TTFT, FP16 agreement, strategy and policy comparisons |
| Fixed-block inference follow-up | Dev | 4 runs x 100 | Select 64 versus 256 tokens after rejecting 16-token blocks |
| INT8 accuracy follow-up | Dev | 2 aligned runs x 300 | Larger FP16-versus-INT8 agreement and accuracy check |
| Timing repetitions | Dev | 8 new runs x 100 | Repetitions 2 and 3 for segmented/document paths on random and Zipf traces |

All synthetic traces use seed 42. `grouped` keeps questions for one article
consecutive, `random` shuffles all real questions, and `zipf` samples article
IDs with exponent 1.1 while cycling through their real questions.

The primary model is `Qwen/Qwen2.5-1.5B-Instruct`. The follow-up manifests
record model revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, prompt version
`quality-mc-v1`, Torch `2.11.0+cu128`, Git revision
`14cdb7cf1697a822cbdb4b9cfd7f5ffce58275c4`, and QuALITY dev checksum
`99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef`.

## Artifact validation

The downloaded follow-up archive was checked before writing this document:

- all 22 JSONL/summary pairs parse, have contiguous indexes, and reproduce
  their stored aggregate fields exactly;
- the 14 confirmation files contain 1,800 records; the eight 10-request smoke
  files contain another 80 records and are excluded from reported estimates;
- all paired fixed-block, INT8, and timing traces have identical
  `(request_index, request_id, article_id)` sequences;
- every confirmation manifest uses the same dataset checksum, seed, model
  revision, prompt version, Torch build, and code revision;
- the INT8 candidate's reference checksum equals the bundled 300-request
  segmented baseline;
- the fixed-block random and Zipf reference checksums equal the corresponding
  original segmented controls recorded in
  [`analysis.json`](generated/inference_confirmation/analysis.json).

Raw result archives remain outside Git, as intended. The complete SHA-256 list
needed to identify this exact follow-up archive is in the provenance appendix.

## Metric conventions

The fair cache-only baseline is segmented inference: it executes the same
`system prefix -> article -> question/options` stages as a cache miss but never
retains article KV. The original full one-forward baseline is an end-to-end
execution comparison, not the denominator for the main cache claim.

`article_token_hit_rate` is the primary reuse metric because it counts restored
article tokens and excludes the small pinned L0 system prompt. The query and
answer options are request-specific and never included in a hit. Root-only L0
matches and `partial_prefix_hit_rate` are therefore not evidence of document
reuse. Atomic document caching has no partial article-text hits by design.

## Complete no-inference matrix

The 108-run test matrix establishes capacity behavior over all 2,128 test
questions and 116 articles. These are geometry-based cache simulations; TTFT,
accuracy, transfer, and dequantization values from this layer must not be
reported as measurements.

### Best article-token hit rate

| Workload | FP16 4 GiB | FP16 8 GiB | INT8 4 GiB | INT8 8 GiB |
|---|---:|---:|---:|---:|
| Grouped | 94.55% | 94.55% | 94.55% | 94.55% |
| Random | 23.44% | 43.78% | 43.71% | 84.89% |
| Zipf | 70.66% | 84.35% | 84.16% | 94.52% |

The near equality of FP16 8 GiB and INT8 4 GiB validates the expected effective
capacity doubling. Grouped order reaches the dataset reuse ceiling, random
traffic churns the cache, and Zipf traffic benefits from a stable hot set.

For random FP16 at 4 GiB with LRU, fixed-block gains only 0.27 percentage
points of article-token hit over document caching, while the measured Python
policy simulator records roughly 9.8 MiB rather than 20 KiB of peak metadata
and far more eviction work. Radix shares only about 0.012% of the budget because
unrelated QuALITY articles rarely have useful common text prefixes. Full trace
details are in [`no_inference_results.md`](no_inference_results.md).

## Initial real-inference confirmation

The table uses the segmented control as the denominator. The full one-forward
means were 6.072 s on random and 6.241 s on Zipf, but those larger end-to-end
speedups include an execution-shape difference and are not attributed solely
to caching.

| Workload | Strategy | Storage | Mean TTFT | Cache-only speedup | Article-token hit | Label agreement |
|---|---|---|---:|---:|---:|---:|
| Random | Document/LRU | Accelerator FP16 | 1.634 s | 1.19x | 20.31% | 100% |
| Random | Fixed-block-16/LRU | Accelerator FP16 | 2.335 s | 0.83x | 20.30% | 100% |
| Random | Radix/LRU | Accelerator FP16 | 1.635 s | 1.19x | 20.26% | 100% |
| Random | Document/LRU | CPU INT8 | 1.456 s | 1.33x | 32.70% | 99% |
| Zipf | Document/LRU | Accelerator FP16 | 0.957 s | 2.19x | 55.93% | 100% vs segmented |
| Zipf | Document/GDSF | Accelerator FP16 | 0.959 s | 2.19x | 55.87% | 100% vs segmented |

Radix provides no aggregate advantage for a one-article-per-request prompt.
LRU and GDSF differ by only 2.1 ms in mean Zipf TTFT in this sample. The initial
100-request INT8 result was exploratory and is superseded by the 300-request
follow-up below.

All FP16 caches preserve the segmented label. The full and segmented Zipf paths
differ on two occurrences of one repeated question; cached FP16 matches the
segmented path, so that discrepancy is an execution-path numerical effect, not
an eviction or restore error.

## Fixed-block granularity

The full no-inference sensitivity trace showed that moving from 16 to 64 and
256 tokens sharply reduced evictions and policy overhead with little hit-rate
loss. The selected 64- and 256-token configurations were then run with real
inference:

| Workload | Block | Mean / p50 / p95 TTFT | Article-token hit | Evictions | Policy/request | Agreement |
|---|---:|---:|---:|---:|---:|---:|
| Random | 16 | 2.335 / 2.764 / 3.726 s | 20.30% | 21,169 | 141.05 ms | 100% |
| Random | 64 | 1.712 / 2.125 / 2.888 s | 20.24% | 5,261 | 32.19 ms | 100% |
| Random | 256 | 1.678 / 2.077 / 2.817 s | 20.06% | 1,293 | 8.10 ms | 100% |
| Zipf | 64 | 1.116 / 0.392 / 2.698 s | 56.27% | 1,956 | 15.64 ms | 100% |
| Zipf | 256 | 1.012 / 0.176 / 2.569 s | 55.79% | 463 | 6.30 ms | 100% |

The 256-token block is the correct generic baseline:

- versus 64 tokens, it reduces mean TTFT by 2.0% on random and 9.3% on Zipf,
  while reducing evictions by about 75%;
- versus 16 tokens on random, it reduces mean TTFT by 28.1%, evictions by
  93.9%, and policy time by 94.3%, for only 0.24 percentage points less hit;
- it is `1.16x` faster than segmented execution on random and `2.07x` on Zipf;
- document caching is still 2.7% faster on random and 5.8% faster on Zipf.

All four new runs preserve the reference label. Zipf/64 has two score-delta
violations at a 0.0625 FP16 threshold but no answer changes; block 256 has no
tolerance violations.

## INT8 accuracy and latency

The larger follow-up compares aligned 300-request random traces:

| Path | Mean / p50 / p95 TTFT | Article hit | Accuracy | Hard accuracy | Agreement |
|---|---:|---:|---:|---:|---:|
| Segmented FP16 | 1.899 / 2.140 / 2.844 s | 0% | 56.33% | 43.59% | Reference |
| Document CPU INT8, 4 GiB | 1.210 / 0.867 / 2.755 s | 41.20% | 55.67% | 42.31% | 99.33% |

INT8 gives a `1.569x` mean speedup, a 36.3% mean TTFT reduction, a 59.5% p50
reduction, and a 3.1% p95 reduction. It changes two answers:

| Index | Request | FP16 -> INT8 | Gold | FP16 margin | INT8 margin | Maximum score delta |
|---:|---|---|---|---:|---:|---:|
| 95 | `20064_CU1CDFL8_6` | C -> A | C | 0.03125 | 0.234375 | 0.203125 |
| 224 | `99915_WLTSM0QE_1` | A -> B | A | 0.296875 | 0.156250 | 0.453125 |

Both changes turn a correct FP16 answer into an incorrect INT8 answer. The
accuracy delta is -0.67 percentage points and the hard-accuracy delta is -1.28
percentage points. The Wilson 95% interval for the observed label-mismatch rate
is approximately 0.18%--2.40%; a two-event sample is not precise enough for a
strong population-level quality claim.

The maximum label-score delta is 1.015625. Of 122 full document restores, 119
exceed the strict 0.0625 FP16 score tolerance, but only two cross an answer
decision boundary. Therefore `119 tolerance violations` does not mean 119
wrong answers; it does show that INT8 is numerically lossy on almost every
restored hit.

`dequant_mean_s = 0.039786` is averaged over all 300 requests, including
misses. The total timed restore work divided by 122 hits is about 97.8 ms per
restore. In the current implementation this timer includes CPU dequantization,
host-to-device copy, dtype conversion, and cache reconstruction. Consequently,
`transfer_mean_s = 0` is an instrumentation convention, not evidence that no
transfer occurred. Separating and accelerating this path is the motivation for
the arena/Triton extension.

## Run-to-run timing stability

Repetitions 2 and 3 use the exact same seed-42 request order as repetition 1.
Each cache run is paired with its segmented control:

| Workload | Segmented means, reps 1/2/3 | Document means, reps 1/2/3 | Median paired speedup (range) | Aggregate mean reduction |
|---|---|---|---:|---:|
| Random | 1.9421 / 2.0399 / 2.0387 s | 1.6339 / 1.6297 / 1.6315 s | 1.250x (1.189--1.252x) | 18.7% |
| Zipf | 2.0967 / 2.0986 / 2.0965 s | 0.9566 / 0.9562 / 0.9554 s | 2.194x (2.192--2.195x) | 54.4% |

The document-cache mean is highly stable: coefficient of variation is 0.13%
on random and 0.06% on Zipf. The Zipf control is also stable. The random
segmented control has 2.8% variation because repetition 1 is faster than the
two later runs; this is why the final claim uses all three paired ratios and
reports their range rather than only the first request-level bootstrap.

Paired cache/control labels match for every request in repetitions 2 and 3.

## Final strategy choices

| Design choice | Selected use | Reason |
|---|---|---|
| Document + LRU + accelerator FP16 | Primary implementation | Best simple latency result; atomic ownership; tiny metadata; exact FP16 labels |
| Fixed-block 256 + LRU | Generic vLLM-like baseline | Competitive after tuning; far less churn than 16/64; still slower than document units |
| Radix + LRU | Architectural comparison only | Partial prefixes exist, but no useful hit or latency advantage for one known article per request |
| GDSF | Skewed/constrained policy comparison | Useful in some Zipf capacity traces; no measured TTFT win over LRU in the selected 100 requests |
| CPU INT8 document cache | Optional capacity mode | Roughly doubles useful capacity and improves median latency, with a measured 2/300 quality cost |

## Validity limits

- QuALITY is document-grounded QA over a bounded stable collection, not
  open-corpus RAG. The request supplies the article ID; there is no retriever in
  this benchmark.
- Test labels are withheld, so the full matrix supports only cache-trace
  claims. Accuracy comes from the smaller dev inference traces.
- Dev accuracy values depend on workload sampling and trace length. In
  particular, Zipf repeats questions and is not a full-split accuracy estimate.
- Only Qwen2.5-1.5B and one CUDA environment have been measured. A second model
  is needed for a generalization claim.
- The original request-level bootstrap intervals capture within-trace request
  variation; the three-run ranges capture a small amount of system variation.
  Neither is a multi-GPU confidence interval.
- CPU INT8 restore phases are not separately instrumented yet, and the current
  cache backend stores Python-owned tensors rather than a preallocated arena.
- The calibrated no-inference prefill model is not observed TTFT and cannot be
  used as if it were a CUDA timing result.

## Remaining work

The empirical 1.5B baseline is now sufficient for the course report. The next
experiments should be narrow rather than another full cross-product:

1. repeat the decisive random and Zipf comparisons with
   `Qwen/Qwen2.5-0.5B-Instruct`, using the same *working-set fraction* as the
   1.5B 4 GiB run;
2. implement the document-owned KV arena and a fused Triton INT8
   restore/dequantization kernel;
3. microbenchmark PyTorch versus Triton restore by tokens and bytes, then rerun
   only segmented, document FP16, and document INT8 end-to-end paths;
4. update the final figures with three-run timing summaries and present INT8 as
   a Pareto tradeoff rather than a lossless optimization.

The concrete sequence is maintained in [`next_steps.md`](next_steps.md).

## Follow-up archive provenance

The formal JSONL inputs checked for this document have these SHA-256 hashes:

| File | SHA-256 |
|---|---|
| `fixed_block_256_random.jsonl` | `3588f1ec19282cf785276304ca1704b3912297b96e7c6ee76a4e7307fadedae2` |
| `fixed_block_256_zipf.jsonl` | `7d17a57119e1b32dfbdcf1021179c23a10afac85d64a0d042b2bd5642bd1111d` |
| `fixed_block_64_random.jsonl` | `d98fb1a92a08295ac6ff2b28f8832b9973c07f71a2c1fd2fc410cc2801869811` |
| `fixed_block_64_zipf.jsonl` | `d3190a5a1c88bfed31416e1cb627d9d68816d99d816851a6c0cfc7f041eb8214` |
| `dev_confirmation_01_segmented_random_fp16.jsonl` | `d8d3799537a7424e564e0a93c07bdd4ba5e3beb070ee0e19c87fafacc9c78b25` |
| `dev_confirmation_02_document_lru_random_int8_4gib.jsonl` | `0c7fadcb9721c356b481276d68cc777e34381815f055d0451f3ea1352d4fa5a2` |
| `dev_confirmation_random_document_lru_fp16_4gib_rep2.jsonl` | `92903cfdf9ce0aaa95284cd50fd2b89392152e6a7fdf161ccb4d1afff5ef328b` |
| `dev_confirmation_random_document_lru_fp16_4gib_rep3.jsonl` | `43a1a3bee765eb584e71dd1ee7150d0260524b659f131af4f01ef1125bd51db4` |
| `dev_confirmation_random_segmented_rep2.jsonl` | `42fc50ba1c4389f2f9c8f37a02f1eb52480f9ced076758cab4e9b070ac3d6fd0` |
| `dev_confirmation_random_segmented_rep3.jsonl` | `3e4a19b27236139c9161ee84f42482ba8a440d77a1523fe68118bd019725c857` |
| `dev_confirmation_zipf_document_lru_fp16_4gib_rep2.jsonl` | `99efd0ff778c538e08b7da68bc4f841778a7387774432865f89b02ed0a45e638` |
| `dev_confirmation_zipf_document_lru_fp16_4gib_rep3.jsonl` | `66a6e49b3e4fb2c64ac0f8b880ea2769624720e355672452af794b68bf219041` |
| `dev_confirmation_zipf_segmented_rep2.jsonl` | `c8eb818799ae178af9f782a09801e9f51b2f74bfab5bb014237493135919056b` |
| `dev_confirmation_zipf_segmented_rep3.jsonl` | `0ce4f259a9a7a20a5841f6c7dcbe73d0736d2f9fdb7f4745069aed667ad92cd7` |
