# Course-report companion

The submission draft is [`../report/main.tex`](../report/main.tex). It uses the
official DLAI 2025/2026 LaTeX style and has two deliberately separated parts:

1. a two-page scientific body for a single-student submission;
2. the mandatory AI-use statement and references.

Timing repetitions and correctness details are consolidated in
[`results.md`](results.md). The committed arena/Triton evidence remains under
[`generated/arena_triton`](generated/arena_triton/README.md).

## Meaning of the comparisons

Document, fixed-block, and radix are local implementations using the same
Transformers runner. The block organization is inspired by vLLM, and the
longest-prefix lookup by SGLang. Neither production engine is executed: their
schedulers, continuous batching, and specialized attention kernels are outside
the comparison. Results therefore characterize this repository's implementations.

The complete-dev suite has one full timing run per configuration. The reported
2--6% differences between cache organizations are observed differences in those
runs. Paired request bootstraps resample aligned requests independently, ignoring
dependence from shared cache state and repeated Zipf questions; their intervals
are descriptive and do not establish repeatable strategy rankings. The earlier
three-run checks cover segmented/document timing on 100 requests, not the full
fixed-block/radix ranking. The later strategy repetitions add three fresh-process
runs per organization/workload on 1,000 requests.
Fixed-256 takes consistently longer than document, while document/radix
latency is very similar. The report includes these results in the body and
links the detailed mean and p90 ranges from the repository; the full-dev table
keeps its original values and revision. These repetitions use one T4 session,
so they do not establish full-dev or multi-GPU rankings.

## Why the title does not say RAG

The original proposal was motivated by RAG: if retrieval repeatedly returns the
same long context, its already-computed KV state could be reused. The specific
target was requests using two or three documents, with the same combinations
recurring across questions. A company legal assistant might repeatedly combine
a statutory provision, an internal policy and a case document; a coding agent
might revisit a small group of files. These are motivating examples, not
measured workloads. The report states that a suitable ready-made public
serving trace was not identified; it does not claim that none exists.

The implemented
benchmark does not contain that retrieval stage. Every QuALITY question already
supplies its exact target `article_id`, so there is no vector index, top-k search,
reranking, or retrieval-quality metric.

This does **not** mean that public RAG or retrieval datasets do not exist. For
example, [KILT](https://arxiv.org/abs/2009.02252) grounds several
knowledge-intensive tasks in one Wikipedia snapshot,
[BEIR](https://arxiv.org/abs/2104.08663) aggregates heterogeneous retrieval
benchmarks, and [RepoBench](https://arxiv.org/abs/2306.03091) evaluates retrieval
of cross-file code context. The gap is more specific: these benchmarks were not
designed as chronological serving traces that jointly preserve repeated request
order, exact serialized prompts, top-k document order, corpus revisions and
invalidation, and a KV-memory budget. Those variables determine cache reuse.
An unordered repeated document set alone is insufficient: exact KV reuse also
requires the same preceding tokens. A later document's KV depends on earlier
documents, so independently prefilling each document and concatenating their
caches would generally change the computation.

This is an intentional controlled setting rather than a hidden omission. A real
RAG experiment would mix cache behavior with retrieval-set overlap, passage
ordering, canonicalization, and retrieval accuracy. The paper therefore studies
the post-routing inference tier:

```text
known article_id -> stable system + article prefix -> KV lookup/restore
                 -> question/options suffix       -> A/B/C/D score
```

The resulting claim is narrower and testable: when an upstream component selects
one stable long document, document boundaries can be useful KV allocation and
eviction units. Representative target environments include coding agents that
repeatedly attach a working set of repository files, legal assistants revisiting
the same contracts or cases, and enterprise assistants over manuals and policies.
These are deployment motivations, not domains evaluated by the project. The
paper does not claim end-to-end RAG speedup.

## Report claims and evidence

| Claim in the two-page body | Evidence boundary |
|---|---|
| Document/LRU gives 1.281x random and 2.741x Zipf mean speedups | All 2,086 requests in each aligned Qwen2.5-1.5B dev trace; paired request bootstrap intervals are reported |
| Document ownership keeps management costs low | Three 1,000-request repetitions find similar document/radix TTFT and lower document metadata/policy cost. Fixed-256 takes a median 1.28% longer on random and 6.54% longer on Zipf, measured relative to document within each pair. |
| GDSF can benefit skewed traffic | The historical full Zipf row improves mean TTFT by 21.4% over document/LRU; it records a known admission-order variant and was not repeated in the organization study. |
| INT8 approximately doubles useful capacity | Complete 2,128-request no-inference test traces: INT8 4 GiB nearly matches FP16 8 GiB |
| INT8 is not lossless | Full random dev: 12/2,086 labels change; full Zipf: 14/911 unique Q&A change after deduplication |
| Zipf occurrence accuracy is not a quality estimate | The 2,086 arrivals contain only 911 unique Q&A; one beneficial INT8 change repeats 32 times, while deduplicated accuracy falls 3/911 |
| Radix partial hits can be misleading | Full Zipf: 301 partial occurrences have a one-token median and contribute only 0.004 percentage points of token-weighted reuse |
| Mean gains do not imply the same p90 gains | Full random/Zipf conditional results show misses still dominate the tail; report p90 values are recalculated from the original timings |
| The arena is a useful negative result | Aligned 100-request tensor/arena comparison with allocation and stale-handle checks |
| Triton improves the isolated restore path | Corrected microbenchmark plus 31 matched online hits; startup and transfer are reported separately |
| The result persists at another scale | Matched-working-set Qwen2.5-0.5B confirmation within the same model family |

The complete-dev suite is now the primary 1.5B evidence. Its 12 JSONLs and
summaries were audited for alignment, checksums, provenance, byte/token bounds,
and exact aggregate reproduction. The earlier 100/300-request runs remain only
for block-size selection, timing repetition, scale, and isolated systems
follow-ups. Smoke runs are excluded from timing evidence; the report-ready
aggregates are retained in [`results.md`](results.md).

## Metric and baseline language

- The main inference table includes mean and p90 TTFT in seconds, alongside
  mean-based speedup and article-token reuse. The segmented control is shown
  for both workloads; descriptive intervals, p50 and full tables are linked
  through the saved result documents.
  p90 is the 90th percentile of request TTFT, not a confidence interval or the
  latency of cache hits alone. It was recalculated from the original JSONLs,
  not estimated from the p95 summaries. The exact values and calculation method
  are summarized in [`results.md`](results.md).
- `article_token_hit_rate` is the primary reuse statistic. It excludes pinned
  L0 and the uncached question/options suffix.
- L0 is the stable system prompt, not the query.
- Atomic document caching has either zero or all article-text tokens; it cannot
  have a partial article-text hit by design.
- The primary control is segmented execution without retained article KV. The
  full one-forward path has a different execution shape and is not the
  denominator for the cache-only claim.
- No-inference rows support occupancy, capacity, hit-rate, metadata, insertion,
  and eviction claims. Their simulated prefill cost is not measured CUDA TTFT.
- CPU INT8 transfer and dequantization are included in real TTFT. Numerical
  tolerance violations are distinct from label mismatches.

## Arena/Triton paragraph: experiment size and source of each number

The former “Systems follow-up” paragraph is titled **“Arena and Triton
experiments (100 requests)”**. It describes completed experiments in this
repository, not future work or timing results published by SGLang/vLLM. These
are earlier, separately versioned 100-request random-dev runs, with seed 42,
Qwen2.5-1.5B, a Tesla T4, and 4 GiB cache budgets. They must not be mistaken
for measurements over the primary 2,086-request suite.

| Measurement | Population and meaning | Frozen source |
|---|---|---|
| Tensor FP16 mean TTFT 1.544 s; arena-64/256 1.723/1.687 s | All 100 requests per run; p90 was recalculated from the saved request timings during report preparation | [run_summaries.csv](generated/arena_triton/run_summaries.csv) |
| INT8 restore 37.45 → 30.98 ms; hit TTFT 94.17 → 86.73 ms | The same 31 cache-hit positions in each PyTorch/Triton run; restore includes transfer, dequantization and assembly, whereas TTFT also includes the rest of the request | [triton_comparison.csv](generated/arena_triton/triton_comparison.csv) |
| Restore speedups 1.137x / 1.167x / 1.243x | Synthetic model-shaped KV at 512 / 2,048 / 8,192 tokens; not QA requests or full-model TTFT | [restore_microbenchmark_runtime_stride.csv](generated/arena_triton/restore_microbenchmark_runtime_stride.csv) |
| Online mean TTFT 1.438 → 1.418 s; Triton 1.445 s including startup | All 100 requests; the 2.671 s JIT warm-up is added separately. The online difference also contains miss/store timing variation | [triton_comparison.csv](generated/arena_triton/triton_comparison.csv) |

The approximate 1,332-request break-even is calculated as
`2.67064 / (0.31 * (0.0374503 - 0.0309815))`. It extrapolates the measured
restore saving at the observed 31% request hit rate; it is not a separately
executed trace or a guaranteed end-to-end crossover. The linked evidence gives
the calculation, timing definitions and CSV source names. The
[provenance record](generated/arena_triton/README.md) identifies measured
revision `710037420045ebc08a4ccd9d28dc1e8ae8b36420` and links the raw-input
hashes; none of these numbers are imported from an external inference engine.

## Before submission

1. Confirm the author name, student number and institutional email in
   [`main.tex`](../report/main.tex).
2. Re-read and personalize the AI-use statement. It must describe the actual
   final workflow, not merely remain as generated boilerplate.
3. Build the PDF and verify that the scientific body ends on page 2 without
   changing the official margins or spacing.
4. Keep the limitations explicit: QuALITY is not retrieval, test labels are
   withheld, Zipf repeats only 911 unique Q&A, and current model evidence covers
   two sizes of one family on one CUDA environment.
5. Recheck every number against [`results.md`](results.md) and the committed
   arena/Triton evidence under [`generated/arena_triton`](generated/arena_triton/README.md).

Compilation and delivery instructions are in
[`../report/README.md`](../report/README.md).
