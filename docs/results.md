# Consolidated experimental results

Status: 7 September 2026. This is the report-ready interpretation of the
completed QuALITY trace, inference, block-size, INT8, and timing-repetition
experiments. The primary Qwen2.5-1.5B evidence is now the complete 2,086-request
dev suite in
[`generated/full_dev_confirmation`](generated/full_dev_confirmation/README.md).
Detailed source tables and figures for the earlier confirmation remain in
[`generated/inference_confirmation`](generated/inference_confirmation/results.md).
The matched-working-set Qwen2.5-0.5B scale confirmation is reported separately
in [`qwen_0.5b_results.md`](qwen_0.5b_results.md). Final allocator and Triton
evidence is frozen in
[`generated/arena_triton`](generated/arena_triton/README.md).

Implementation note (5 September 2026): the later
[radix access-accounting fix](code_quality.md#scoped-cleanup-and-radix-access-correction-5-september-2026)
removes an extra lookup during insertion. Radix values below remain evidence
for the recorded historical revisions, not measurements of the corrected
implementation. Document/fixed-block and arena/Triton algorithms are unchanged.
The full-dev GDSF row also predates a known object-cache ordering correction:
incoming priority is computed before eviction advances the aging clock. It is
reported as the behavior of that recorded variant and one Zipf trace, not as a
canonical or universal GDSF result.

## Executive result

The evidence supports document-aware KV caching for repeated long-document QA.
Over all 2,086 dev requests, a 4 GiB accelerator-FP16 document/LRU cache reduces
mean TTFT relative to the fair segmented no-document-cache control by 21.9% on
random traffic (`1.281x`) and 63.5% on Zipf traffic (`2.741x`). The paired
request-level bootstrap intervals are `[1.250x, 1.313x]` and
`[2.594x, 2.903x]`.

The observed ordering differs by workload. Document/LRU has the lowest measured
accelerator-FP16 mean TTFT on random traffic: its mean is 2.32% below tuned
fixed-block-256 and 5.80% below radix at almost the same token hit rate. On the
seed-42 Zipf trace, radix/LRU is 5.42% faster than document/LRU, while
document/GDSF retains more hot articles and reaches `3.487x` versus segmented.
Known document boundaries therefore provide an efficient allocation unit, but
neither the organization nor the eviction policy is a universal winner.
These are local implementations on one Transformers runner, inspired by vLLM's
block organization and SGLang's radix lookup. Neither production engine is
benchmarked. The later three-repetition, 1,000-request comparison now finds
very similar document/radix latency and consistently slower fixed-256 timing.
It uses the corrected radix implementation and is reported separately in
[the repetition results](generated/strategy_repetitions/README.md); it does not
turn the historical full-dev ordering into a universal ranking.

CPU INT8 approximately doubles capacity. On the full random permutation it
reaches a 44.67% article-token hit and `1.578x` mean speedup. It changes 12/2,086
labels (`99.425%` agreement); six changes help and six hurt, so overall accuracy
remains exactly 1,210/2,086 while hard accuracy falls by two answers. On Zipf,
occurrence weighting misleadingly suggests an accuracy increase. Deduplicating
the 911 sampled Q&A gives 496/911 versus the segmented 499/911, a -0.329-point
change. INT8 is therefore a measured memory/latency/quality tradeoff, not a
lossless or accuracy-improving mode.

The optional systems follow-up reaches a more nuanced conclusion. A document-
owned page arena enforces deterministic lifetime and byte accounting but is
9--12% slower than ordinary tensor storage because Transformers still
reassembles a contiguous cache. The corrected Triton INT8 kernel reduces hit-
only restore time by 17.3% and dequantization by 73.4%, but host transfer
dominates and a 2.671-second one-time JIT warm-up requires a long-lived server
to amortize.

## Research questions and answers

| Question | Answer from the current evidence |
|---|---|
| Does cross-request article-KV reuse reduce TTFT? | Yes: the full-dev document/LRU speedup is `1.281x` on random and `2.741x` on Zipf; document/GDSF reaches `3.487x` on the selected Zipf trace. |
| When do policies differ? | Locality and capacity dominate. LRU is the simple random default; GDSF improves mean TTFT by 21.4% over document/LRU on this skewed trace, but this is one seed rather than a universal ranking. |
| Does INT8 increase useful capacity? | Yes. INT8 4 GiB closely matches FP16 8 GiB in the full trace simulation. |
| Is INT8 accuracy-neutral? | No strict lossless claim is justified: 12/2,086 random labels and 14/911 unique Zipf Q&A change, even though the full-random net accuracy happens to be zero. |
| Does document-aware caching beat generic prefix structures? | The later three-run comparison finds very similar document/radix TTFT with much lower document metadata and management time. Fixed-256 is slower in all paired repetitions. The historical full-dev Zipf radix lead is not supported by this current 1,000-request comparison. |
| Is offline precomputation amortized? | Not yet answered experimentally. Calibrated prefill estimates are available, but no measured online-versus-offline prefill comparison should be claimed. |
| Does a preallocated page arena improve this Transformers path? | No. It strengthens ownership and fragmentation accounting, but transient contiguous reconstruction makes it slower than tensor storage. |
| Does Triton improve CPU-INT8 restore? | Yes on the hit path: restore is 1.209x faster and dequantization 3.754x faster. Startup-amortized TTFT is not a win at only 100 requests. |

## Experimental layers

The project uses two separate dataset roles and does not mix their metrics:

| Layer | Split | Requests | Purpose |
|---|---|---:|---|
| Complete no-inference matrix | Test | 108 runs x 2,128 | Cache capacity, hit rate, policy, and metadata behavior without labels or model forwards |
| Initial inference confirmation | Dev | 10 aligned runs x 100 | Fair TTFT, FP16 agreement, strategy and policy comparisons |
| Fixed-block inference follow-up | Dev | 4 runs x 100 | Select 64 versus 256 tokens after rejecting 16-token blocks |
| INT8 accuracy follow-up | Dev | 2 aligned runs x 300 | Larger FP16-versus-INT8 agreement and accuracy check |
| Timing repetitions | Dev | 8 new runs x 100 | Repetitions 2 and 3 for segmented/document paths on random and Zipf traces |
| Qwen2.5-0.5B scale confirmation | Dev | 6 aligned runs x 100 | Matched-working-set check of the decisive random and Zipf comparisons |
| Arena/Triton confirmation | Dev | 6 aligned runs x 100 | Tensor versus page-arena FP16 and PyTorch versus Triton CPU-INT8 restore |
| Corrected restore microbenchmark | Synthetic model geometry | 2 repetitions x 6 rows | Isolate transfer, dequantization, restore throughput, and numerical parity at 512/2,048/8,192 tokens |
| Complete 1.5B dev inference | Dev | 12 aligned runs x 2,086 | Primary full-split random accuracy plus random/Zipf latency, strategy, policy, CPU storage, and INT8 correctness |
| Shortlisted strategy repetitions | Dev | 18 cached runs x 1,000 + 2 references x 1,000 | Three fresh-process repetitions per organization/workload after simulation-based selection; current radix implementation, schema v4 |

All synthetic traces use seed 42. `grouped` keeps questions for one article
consecutive, `random` shuffles all real questions, and `zipf` samples article
IDs with exponent 1.1 while cycling through their real questions.

The primary model is `Qwen/Qwen2.5-1.5B-Instruct`. The complete-dev manifests
record model revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, prompt version
`quality-mc-v1`, Torch `2.11.0+cu128`, Git revision
`89acf36a603adf7dacd5c8dd0b32a2967f4bbee2`, and QuALITY dev checksum
`99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef`.
Earlier follow-up and systems archives retain their own recorded revisions.

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

The complete-dev archive was audited separately: all 12 JSONLs contain 2,086
contiguous rows, all paired traces and reference checksums align, all stored
summaries recompute exactly, all byte/token invariants hold, and all runs share
one dataset/model/prompt/code/hardware provenance. The source zip SHA-256 is
`204a85e7bf64f52a912f018c65985a0de971d04be77de0d66dfbbb0e2d7ddf33`.
Exact run hashes and the unique-Q&A Zipf analysis are in
[`generated/full_dev_confirmation/README.md`](generated/full_dev_confirmation/README.md).

The earlier block-size, 300-request INT8, and timing-repetition artifacts use
result schema `quality-kv-v2`. The primary complete-dev and final arena/Triton
archives use `quality-kv-v3`, which tokenizes once outside the measured
model-forward interval and reports combined `restore_s` plus a separate
`store_s`. Current code emits `quality-kv-v4` after correcting arena metadata
semantics. Do not merge request-level or summary rows across schemas; every
claim here is tied to the explicitly named archive and revision.

The final arena/Triton archive is entirely `quality-kv-v3`. Its six 100-request
JSONLs have aligned trace positions and a common segmented reference. The
analyzer verified the dataset, model, code, and hardware provenance; exact
input hashes are stored in
[`analysis.json`](generated/arena_triton/analysis.json). The two corrected
microbenchmark repetitions have complete Tesla-T4/CUDA/Triton manifests and
exact PyTorch output parity. Raw JSONLs remain outside Git; only the curated
tables, figures, hashes, and manifests are checked in.

To audit the final systems phase, use this order:

| Check | Curated artifact |
|---|---|
| Machine, dataset, model, schema, and raw-input provenance | [`generated/arena_triton/README.md`](generated/arena_triton/README.md) and [`analysis.json`](generated/arena_triton/analysis.json) |
| Per-path TTFT, hit rate, allocator, and timer aggregates | [`run_summaries.csv`](generated/arena_triton/run_summaries.csv) |
| Paired speedups and bootstrap intervals | [`fair_speedups.csv`](generated/arena_triton/fair_speedups.csv) |
| FP16/INT8 answer agreement | [`correctness.csv`](generated/arena_triton/correctness.csv) and [`mismatch_details.json`](generated/arena_triton/mismatch_details.json) |
| Matched PyTorch/Triton hit path and startup amortization | [`triton_comparison.csv`](generated/arena_triton/triton_comparison.csv) |
| Corrected isolated restore repetitions | [`restore_microbenchmark_runtime_stride.csv`](generated/arena_triton/restore_microbenchmark_runtime_stride.csv) and [`restore_microbenchmark_rep1.csv`](generated/arena_triton/restore_microbenchmark_rep1.csv) |

The human-readable synthesis is [`arena_triton.md`](arena_triton.md). The
generated summary in
[`generated/arena_triton/results.md`](generated/arena_triton/results.md) is
kept unedited so it can be regenerated directly from the ignored raw JSONLs.

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

## Complete full-dev inference confirmation

This is now the primary Qwen2.5-1.5B result. Each of the 12 aligned runs serves
2,086 requests with a 4 GiB cache budget on one Tesla T4. Random is a
permutation of every labeled dev question; Zipf is a repeated synthetic trace
of the same length and is used for locality and policy behavior.

| Workload | Path | Mean / p50 / p95 TTFT | Article-token hit | Speedup vs segmented (95% CI) | Label changes |
|---|---|---:|---:|---:|---:|
| Random | Segmented FP16 | 1.843 / 2.107 / 2.889 s | 0% | Reference | n/a |
| Random | Document/LRU accelerator FP16 | 1.439 / 1.598 / 2.828 s | 22.94% | 1.281x [1.250, 1.313] | 0 |
| Random | Fixed-256/LRU accelerator FP16 | 1.474 / 1.558 / 2.877 s | 23.15% | 1.251x [1.223, 1.281] | 1 |
| Random | Radix/LRU accelerator FP16 | 1.528 / 1.752 / 2.928 s | 22.77% | 1.206x [1.178, 1.236] | 0 |
| Random | Document/LRU CPU FP16 | 1.574 / 1.773 / 2.994 s | 22.94% | 1.171x [1.144, 1.200] | 0 |
| Random | Document/LRU CPU INT8 + Triton | 1.168 / 0.786 / 2.908 s | 44.67% | 1.578x [1.520, 1.643] | 12 |
| Zipf | Segmented FP16 | 2.208 / 2.323 / 2.733 s | 0% | Reference | n/a |
| Zipf | Document/LRU accelerator FP16 | 0.806 / 0.068 / 2.716 s | 65.38% | 2.741x [2.594, 2.903] | 0 |
| Zipf | Fixed-256/LRU accelerator FP16 | 0.863 / 0.185 / 2.742 s | 65.27% | 2.558x [2.437, 2.693] | 3 occurrences / 2 Q&A |
| Zipf | Radix/LRU accelerator FP16 | 0.762 / 0.068 / 2.546 s | 65.23% | 2.898x [2.745, 3.068] | 0 |
| Zipf | Document/GDSF accelerator FP16 | 0.633 / 0.065 / 2.539 s | 71.46% | 3.487x [3.280, 3.721] | 0 |
| Zipf | Document/LRU CPU INT8 + Triton | 0.455 / 0.087 / 2.563 s | 81.66% | 4.851x [4.503, 5.251] | 57 occurrences / 14 Q&A |

The mean result is much stronger than the tail result because a cache miss
still performs almost the complete article prefill. Random document/LRU lowers
p95 by only 2.1%; random INT8 is 0.6% worse at p95 despite its 36.6% mean
reduction. On Zipf, the best accelerator p95 reduction is 7.1%, versus a 71.3%
mean reduction for GDSF. The report therefore does not translate mean TTFT into
a blanket tail-latency claim.

The random segmented model answers 1,210/2,086 questions correctly (58.006%)
and 521/1,065 hard questions correctly (48.920%). Document accelerator FP16,
radix FP16, and CPU FP16 preserve every label; fixed-256 changes one wrong
answer into a correct answer. INT8 changes 12 labels: six correct-to-wrong and
six wrong-to-correct, leaving overall accuracy unchanged while hard accuracy
falls by 2/1,065 (-0.188 percentage points). Its 903 strict score-tolerance
violations show that answer agreement does not make the quantizer numerically
lossless.

### Deduplicated Zipf view

The Zipf trace has 911 unique Q&A from 110 articles. Occurrence-weighted INT8
accuracy rises by 0.91 points because one baseline-wrong question repeats 32
times and becomes correct. With one modal prediction per unique Q&A, accuracy
instead changes from 499/911 (54.775%) to 496/911 (54.446%), and hard accuracy
from 244/509 to 242/509. The 14 unique answer changes comprise seven
correct-to-wrong, four wrong-to-correct, and three wrong-to-different-wrong.
No INT8 accuracy improvement is claimed.

For cache breadth, the report uses `unique-Q&A cache coverage@90%`: the fraction
of the 911 sampled Q&A for which at least one occurrence restores 90% or more
of its article. Coverage is 38.86% for document/LRU, 39.41% for fixed-256,
38.75% for radix, 39.41% for document/GDSF, and 60.04% for CPU INT8. Radix's
any-positive-reuse coverage is 64.00%, but this is inflated by one-token
matches; its useful 90%-coverage is not higher. This is cache coverage, not
retrieval recall, because the request already supplies the article ID.

All exact tables, conditional hit/miss TTFT, partial-prefix distributions,
resource measurements, correctness accounting, and raw input hashes are in
[`generated/full_dev_confirmation/README.md`](generated/full_dev_confirmation/README.md).

## Completed strategy repetitions (7 September 2026)

The simulation shortlist was followed by selected inference, then three
fresh-process runs of each GPU FP16 organization on the first 1,000 requests
of the seed-42 random and Zipf traces. LRU, a 4 GiB tensor budget and
Qwen2.5-1.5B are held constant; execution order rotates. All 20 planned runs
finished in 8.27 hours, including two single segmented correctness references.

| Workload | Cache | Median mean TTFT (min–max), s | Median p90, s | Article-token hit |
|---|---|---:|---:|---:|
| Random | Document | 1.731 (1.730–1.735) | 3.015 | 22.00% |
| Random | Fixed-256 | 1.754 (1.752–1.757) | 3.049 | 22.57% |
| Random | Radix | 1.735 (1.731–1.735) | 3.020 | 21.99% |
| Zipf | Document | 0.915 (0.913–0.915) | 2.819 | 63.43% |
| Zipf | Fixed-256 | 0.973 (0.970–0.975) | 2.834 | 63.40% |
| Zipf | Radix | 0.917 (0.916–0.917) | 2.819 | 63.44% |

Paired within repetition, fixed-256 takes a median 1.28% longer than document
on random and 6.54% longer on Zipf, with the same ordering in all three runs.
Radix takes only 0.02% and 0.19% longer, respectively; p90 ordering changes
between runs. This supports similar document/radix latency rather than a
meaningful latency winner. Estimated document metadata is about 0.02 MiB
versus 5.2–5.5 MiB, and its policy time is much lower. Lower management cost
is the clearer document advantage on this workload.

All 18,000 cached observations preserve the segmented reference labels.
Some fixed-block/radix scores exceed the numerical tolerance without changing
labels. All byte bounds, trace alignment, input hashes and completion checks
pass; regenerated tables match the downloaded analysis exactly.

These schema-v4 runs use revision `68e7080` with corrected radix accounting.
The shorter traces and changed environment prevent attributing differences
from the historical full-dev suite to the fix alone. There is no request
bootstrap or equivalence test, and the single references are not repeated
timing controls. Full-dev remains the main accuracy evidence. Exact per-run
values, p90 ranges, audit and reproduction commands are in
[generated/strategy_repetitions](generated/strategy_repetitions/README.md).

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

## Earlier real-inference confirmation

These smaller runs remain historical sensitivity and repeatability evidence;
the complete-dev section above supersedes them for primary 1.5B point
estimates. The table uses the segmented control as the denominator. The full one-forward
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

In the frozen 300-request archive, `dequant_mean_s = 0.039786` is averaged over all 300 requests, including
misses. The total timed restore work divided by 122 hits is about 97.8 ms per
restore. In that archived implementation this timer includes CPU dequantization,
host-to-device copy, dtype conversion, and cache reconstruction. Consequently,
`transfer_mean_s = 0` is an instrumentation convention, not evidence that no
transfer occurred. Separating and accelerating this path is the motivation for
the now-implemented arena/Triton extension. New backend rows use separately
instrumented transfer, device dequantization, assembly, and combined restore
timers and must not be merged silently with the archive.

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
| Document + LRU + accelerator FP16 | Primary random-traffic implementation | Lowest observed random mean TTFT; atomic ownership; small estimated metadata; exact labels |
| Fixed-block 256 + LRU | Local baseline inspired by vLLM's block organization | Competitive after tuning, but much more churn and higher observed mean TTFT than document/LRU; no vLLM engine benchmark |
| Radix + LRU | Architectural comparison | Slower on random, faster than document/LRU on this Zipf trace; most non-full prefixes are only one token |
| GDSF | Skewed-workload policy | Best accelerator result on this one Zipf trace; workload-dependent, not a universal win |
| CPU INT8 document cache | Capacity mode | Roughly doubles useful capacity and improves mean/median latency, with 12/2,086 random label changes |

## Matched-working-set 0.5B confirmation

The six-run Qwen2.5-0.5B experiment holds the cache at the same 22.80% fraction
of the model's FP16 article-KV working set as the 1.5B 4 GiB experiment. It
reproduces the main result: document LRU gives `1.206x` cache-only speedup on
random traffic and `2.154x` on Zipf, with article-token hit rates of 20.31% and
55.93%. All four FP16 cached paths preserve every segmented-reference label.

On random traffic, document caching is 4.6% faster than fixed-block 256 and
0.4% faster than radix in mean TTFT. Fixed-block has 1,293 evictions versus 56
for document caching and spends about 51 times as much policy time per request.
Radix is close in latency but uses about 307 times as much peak metadata. The
full protocol, confidence intervals, correctness table, limitations, and input
hashes are in [`qwen_0.5b_results.md`](qwen_0.5b_results.md).

## Arena and Triton systems follow-up

The aligned 100-request arena experiment is complete. Against a 1.941-second
segmented control, tensor FP16, arena-64, and arena-256 have mean TTFTs of
1.544, 1.723, and 1.687 seconds. Both arena modes preserve every FP16 label and
all allocation invariants, but neither beats tensor storage because the current
Transformers attention path still reconstructs contiguous caches. Arena-64
uses less tail padding and retains the 20.31% tensor hit rate; arena-256 reduces
page-operation overhead but its 114.35 MiB peak tail waste lowers hit rate to
18.19%.

The corrected restore microbenchmark gives 1.137x, 1.167x, and 1.243x Triton
restore speedups at 512, 2,048, and 8,192 tokens, with exact output parity. On
the 31 matched cache hits, Triton reduces mean dequantization from 8.06 to 2.15
ms and complete restore from 37.45 to 30.98 ms. The PyTorch and Triton INT8
paths produce identical A/B/C/D scores and the same one FP16 label mismatch.

Online Triton mean TTFT is 1.418 seconds versus 1.438 for PyTorch, but this
small all-request difference also contains run-to-run miss/store variation.
The isolated claim is therefore the hit restore improvement. The separately
reported 2.671-second JIT warm-up makes Triton about 0.49% slower when
amortized over only 100 requests; the observed hit-restore saving breaks even
after roughly 1,332 requests at a 31% hit rate. Full tables, hashes, and
interpretation are in [`arena_triton.md`](arena_triton.md).

## Validity limits

- QuALITY is document-grounded QA over a bounded stable collection, not
  open-corpus RAG. The request supplies the article ID; there is no retriever in
  this benchmark.
- Test labels are withheld, so the no-inference test matrix supports only
  cache-trace claims. Full-split accuracy comes from the 2,086-request random
  dev inference trace.
- Dev accuracy values depend on workload sampling and trace length. In
  particular, Zipf repeats questions and is not a full-split accuracy estimate.
- Only two sizes from the same Qwen2.5 family and one CUDA environment have
  been measured. This supports scale robustness within the family, not a
  cross-architecture or multi-GPU generalization claim.
- The full-dev paired request bootstrap resamples aligned requests independently.
  It ignores dependence from shared cache state and repeated Zipf questions;
  these descriptive intervals do not establish repeatable strategy rankings.
  The later three-run comparison covers all three organizations on 1,000-request
  traces in one T4 session. It supports descriptive timing comparisons on those
  traces, not statistical significance, full-dev ranking or multiple GPUs.
- All organizations use local cache implementations on the same Transformers
  runner. The study does not execute the vLLM or SGLang engines, their schedulers,
  continuous batching, or specialized attention kernels.
- The frozen 300-request INT8 result predates separate restore-stage
  instrumentation. The complete-dev v3 INT8 rows now provide the primary
  correctness estimate; the matched 100-request systems comparison remains the
  evidence that isolates PyTorch versus Triton restore.
- Triton startup is reported separately from online TTFT. The all-request
  PyTorch/Triton difference cannot be assigned entirely to the kernel because
  independent runs also differed in miss/store time.
- The calibrated no-inference prefill model is not observed TTFT and cannot be
  used as if it were a CUDA timing result.

## Experimental status and optional future work

The complete 1.5B dev suite, matched-working-set 0.5B check, page-arena study,
and corrected Triton comparison are complete. No additional experiment is
required before writing the course report. The original length-specialized
Triton trace remains diagnostic evidence and is not mixed into final tables.

Optional extensions are pinned/asynchronous CPU transfer with stream-safe
lifetime management, page-table-aware attention that consumes arena locations
without contiguous reconstruction, and replication on another GPU or model
family. These are new research phases, not missing validation for the current
claims. The finished-versus-optional boundary is maintained in
[`next_steps.md`](next_steps.md).

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
