# Qwen2.5-0.5B matched-working-set results

Status: 31 August 2026. This document freezes the interpretation and
provenance of the completed Qwen2.5-0.5B confirmation. The reproducible matrix
is encoded in
[`../configs/qwen_0.5b_confirmation.json`](../configs/qwen_0.5b_confirmation.json).

## Executive result

The smaller-model experiment reproduces the main 1.5B result at a matched
cache-capacity regime. On 100 aligned QuALITY dev requests, document caching
with LRU reduces mean TTFT relative to segmented inference by 17.1% on random
traffic and 53.6% on Zipf traffic. This corresponds to cache-only speedups of
`1.206x` and `2.154x`, respectively.

On the random trace, document caching has the lowest mean TTFT of the three
cache organizations. It is 4.6% faster than fixed-block-256 and 0.4% faster
than radix. The document and radix timings are therefore effectively close in
this sample, but document caching obtains that result with much less metadata
and policy work. All FP16 cache paths preserve every answer label produced by
their aligned segmented control.

The result supports robustness across two sizes in the Qwen2.5 family. It is
not evidence of generalization to unrelated model architectures or hardware.

## Experimental protocol

| Field | Value |
|---|---|
| Dataset | Official HTML-stripped QuALITY v1.0.1 dev split |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Model revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Prompt | `quality-mc-v1` |
| Workloads | Seed-42 random and Zipf-1.1 |
| Requests | 100 aligned requests per run |
| Storage | CUDA-resident FP16 |
| Policy | LRU for every cached path |
| Result schema | `quality-kv-v3` |
| Timing scope | `model-forward-excludes-tokenization-v1` |
| Torch | `2.11.0+cu128` |
| Code revision | `cc2584872ccfc8e0b2ddfe26e3cf102b8fe3decd` |

The suite contains six runs:

| Workload | Execution paths |
|---|---|
| Random | Segmented control, document, fixed-block 256, and radix |
| Zipf | Segmented control and document |

Segmented inference is the fair cache-only baseline. It executes the same
stable-system-prefix, article, and question/options stages as a cache miss, but
does not retain article KV between requests. The suite intentionally does not
contain a full one-forward control, so no end-to-end speedup is reported.

L0 contains only the stable system-prompt KV. It is pinned, but it is excluded
from `article_token_hit_rate`; the query and answer options are never cached.

### Matched capacity

The cached runs use `22.80130165664403%` of the model's complete FP16 article-KV
working set. This is the fraction occupied by 4 GiB in the frozen 1.5B
experiment:

```text
4,294,967,296 / 18,836,500,480 = 22.80130165664403%
```

For Qwen2.5-0.5B, the measured working set is `8,072,785,920` bytes and the
resolved budget is `1,840,700,269` bytes, or approximately 1.714 GiB. The
comparison therefore holds relative capacity constant instead of giving the
smaller model an easier fixed 4 GiB budget.

## Latency and reuse

| Workload | Path | Mean / p50 / p95 TTFT | Speedup (paired 95% CI) | Mean reduction | Article-token hit | Full-document hit |
|---|---|---:|---:|---:|---:|---:|
| Random | Segmented | 0.952 / 1.050 / 1.335 s | Reference | Reference | 0% | 0% |
| Random | Document LRU | 0.789 / 0.985 / 1.363 s | 1.206x [1.095, 1.347] | 17.1% | 20.31% | 19% |
| Random | Fixed-block 256 LRU | 0.827 / 1.012 / 1.389 s | 1.150x [1.054, 1.275] | 13.1% | 20.06% | 0% |
| Random | Radix LRU | 0.792 / 1.011 / 1.366 s | 1.201x [1.092, 1.344] | 16.8% | 20.26% | 19% |
| Zipf | Segmented | 1.017 / 1.064 / 1.251 s | Reference | Reference | 0% | 0% |
| Zipf | Document LRU | 0.472 / 0.045 / 1.259 s | 2.154x [1.783, 2.710] | 53.6% | 55.93% | 55% |

The paired bootstrap intervals are computed over requests within this one
execution of each path. They quantify within-trace request variation, not
run-to-run, GPU-to-GPU, or multi-model uncertainty.

The mean and median improve, but p95 does not. Relative to segmented inference,
p95 is about 2.1% higher for random/document, 4.1% higher for
random/fixed-block, 2.3% higher for random/radix, and 0.6% higher for
Zipf/document. Hits create a fast mode and drive the mean and median gains;
cache misses remain in the high-latency tail. This experiment therefore does
not support a tail-latency improvement claim.

## Strategy-management cost

The following values compare the three random-workload caches under the same
trace and capacity:

| Strategy | Evictions | Lookup/request | Restore/request | Store/request | Policy/request | Peak metadata |
|---|---:|---:|---:|---:|---:|---:|
| Document | 56 | 0.009 ms | 0.252 ms | 0.524 ms | 0.159 ms | 17,614 B |
| Fixed-block 256 | 1,293 | 0.374 ms | 0.547 ms | 31.698 ms | 8.086 ms | 5,717,134 B |
| Radix | 58 | 0.366 ms | 0.492 ms | 0.468 ms | 1.726 ms | 5,415,001 B |

Fixed-block-256 restores almost the same fraction of article tokens as the
document cache, but it performs 23 times as many evictions, uses about 325
times as much peak metadata, spends about 60 times as long in storage work per
request, and spends about 51 times as long in policy management. Its block
granularity does not translate into a useful hit-rate advantage on this trace.

Radix obtains a partial article-text hit on 29% of requests, compared with zero
partial article-text hits for the atomic document cache. Nevertheless,
its aggregate article-token hit rate and mean TTFT do not improve. It uses
about 307 times the document cache's metadata and about 11 times its policy
time. This supports keeping radix as an architectural comparison rather than
the preferred organization for one known, unrelated article per prompt.

`full_document_hit_rate = 0` for fixed-block does not mean it restored no
article text. Its 20.06% article-token hit is composed of partial block-level
matches. Conversely, document caching has no partial article-text hit by
design: an article is restored atomically or missed.

## Correctness

| Workload | Cache path | Label agreement | Accuracy | QuALITY-hard accuracy | Maximum A/B/C/D logit delta |
|---|---|---:|---:|---:|---:|
| Random | Document LRU FP16 | 100/100 | 42.0% | 35.7% | 0.000000 |
| Random | Fixed-block 256 LRU FP16 | 100/100 | 42.0% | 35.7% | 0.062500 |
| Random | Radix LRU FP16 | 100/100 | 42.0% | 35.7% | 0.046875 |
| Zipf | Document LRU FP16 | 100/100 | 51.0% | 51.8% | 0.000000 |

There are zero label mismatches and zero accuracy deltas relative to each
aligned segmented baseline. The fixed-block and radix score differences stay
within the configured FP16 tolerance and never change the selected label.

These accuracy values describe the sampled 100-request traces. Zipf repeats
questions according to its synthetic popularity distribution, so its 51%
accuracy is not a full-dev-split estimate and must not be compared with random
accuracy as if both were independent benchmark scores.

This suite contains accelerator FP16 only. It does not revise the separate
1.5B CPU-INT8 finding of 298/300 label agreement and must not be used to make a
0.5B INT8 quality claim.

## Comparison with the frozen 1.5B result

The new traces use the same seed, request order, prompt, and matched fraction
of the FP16 working set. The cache-only conclusions are close:

| Comparison | Qwen2.5-1.5B | Qwen2.5-0.5B |
|---|---:|---:|
| Document/random article-token hit | 20.31% | 20.31% |
| Document/random speedup | 1.250x median over three runs | 1.206x, one run |
| Fixed-block-256/random speedup | 1.16x | 1.150x |
| Document/Zipf article-token hit | 55.93% | 55.93% |
| Document/Zipf speedup | 2.194x median over three runs | 2.154x, one run |

This confirms that the qualitative effect is not unique to the 1.5B parameter
count: locality controls reusable work, and document ownership removes
management overhead without sacrificing useful token reuse. However, the 1.5B
archive uses result schema v2 while the 0.5B archive uses v3. Absolute TTFT
values must not be concatenated or directly compared across schemas; only
within-suite speedup ratios and consistently defined hit metrics are used
above.

## Artifact validation

Before writing this document, the downloaded archive was checked for the
following invariants:

- all six confirmation JSONL files are present and contain 100 contiguous
  request indexes, for 600 request rows in total;
- random and Zipf cached paths align position-by-position with their segmented
  references, including repeated Zipf request IDs;
- all manifests share the same dataset checksum, prompt version, model
  revision, code revision, seed, timing scope, and result schema;
- all FP16 comparisons have 100% reference-label agreement;
- every request respects its tensor-byte budget and token-hit bounds; and
- the analyzer reports `traces_aligned = true` and
  `segmented_invariants = true`.

The raw archive remains outside Git according to the repository artifact
policy. The analyzer outputs have these identifying hashes:

| Artifact | SHA-256 |
|---|---|
| `analysis.json` | `fcda1e3561690d68272286d9b926b215abd37e12875a682abcdbc547e8891d24` |
| `results.md` | `8f9ea3d3a0b20a0fafa8f0178aac7c700dba415c80612fcf9645abd0977d1209` |

The six formal JSONL inputs recorded by the analyzer are:

| Run | SHA-256 |
|---|---|
| Segmented random | `b05dc0ebc3068f12295bc714440ee6a9cb1a42ecf4aaea1160bff0cb6c80ec48` |
| Document random | `7f8817ce72d56dc4b2026e8b2d7ffca13c702e45c4157beecfad10b6940c0aba` |
| Fixed-block-256 random | `c330a4685a4fe883edd7c9bcbef4806f8b1ecc2161657602b571b90d3faeeb34` |
| Radix random | `8e838a321438c3a2a8b0f12c6b7c0c6b932eb91eb3481dc9be163fb0d9317699` |
| Segmented Zipf | `2220261d3089a1e0a8c0447d72886ad9f25e2f0be03335c2255de5939d00e0da` |
| Document Zipf | `6a9f80f82c0e4330709c5c8fed3ac56690943e805e16b51fc3b66f5fabb1df21` |

The dataset checksum is
`99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef`.

## Decision and subsequent completed work

This confirmation strengthens the current design decision:

- document + LRU + accelerator FP16 remains the primary cache;
- fixed-block 256 remains the tuned generic/vLLM-like baseline;
- radix remains an architectural comparison whose extra structure is not
  rewarded by this one-article workload; and
- CPU INT8 remains a separate capacity/latency/quality tradeoff established by
  the larger 1.5B experiment.

The next implementation step was not another model-scale matrix. The project
subsequently completed a document-owned KV arena, corrected Triton INT8
restore/dequantization path, focused restore microbenchmark, and aligned
100-request comparison against PyTorch. Those results do not alter this 0.5B
scale conclusion; they are separately versioned in
[`arena_triton.md`](arena_triton.md) and
[`generated/arena_triton`](generated/arena_triton/README.md).
