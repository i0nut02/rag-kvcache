# Completed follow-ups and remaining work

The 1.5B empirical baseline is now frozen. Fixed-block granularity, the larger
INT8 correctness check, run-to-run timing repetitions, and the matched-capacity
0.5B confirmation are complete. The full interpretation is in
[`results.md`](results.md); this page separates finished evidence from the
remaining implementation plan.

## Decisions now supported by data

- Use atomic document caching with LRU and accelerator FP16 as the primary
  system.
- Use 256-token fixed blocks as the tuned generic/vLLM-like comparison. Do not
  use the 16-token result as the only fixed-block baseline.
- Keep radix caching as an architectural comparison, not as the proposed
  winner for one-article QuALITY prompts.
- Present GDSF as a workload-dependent policy for skewed, constrained traces.
  It does not beat LRU in the selected 100-request CUDA timing comparison.
- Present CPU INT8 as a Pareto tradeoff. It is faster and approximately doubles
  capacity, but it is not label-lossless.

## Completed follow-up 1: fixed-block granularity

The full no-inference sensitivity sweep and selected real-inference runs both
favor 256 tokens. Relative to 64 tokens, block 256 reduces mean TTFT by 2.0% on
random and 9.3% on Zipf traffic while reducing evictions by about 75%. On the
random inference trace it reduces TTFT by 28.1% and evictions by 93.9% relative
to the original 16-token cache.

Block 256 reaches `1.16x` cache-only speedup on random and `2.07x` on Zipf, but
document caching remains 2.7% and 5.8% faster, respectively. All selected
fixed-block labels agree with the segmented reference.

Reproduction commands:

```bash
python experiments/run_quality.py matrix configs/fixed_block_sensitivity.json \
  --profile full --execute --resume
python experiments/run_quality.py matrix configs/fixed_block_inference.json \
  --profile confirmation --execute --resume
```

## Completed follow-up 2: INT8 accuracy

The 300-request random comparison measures segmented FP16 at 1.899 s mean TTFT
and document CPU INT8 at 1.210 s. INT8 gives a `1.569x` speedup and 41.20%
article-token hit rate. Label agreement is 298/300 (`99.33%`); both changed
answers were correct in FP16 and incorrect in INT8. Overall accuracy changes
from 56.33% to 55.67%, and QuALITY-hard accuracy from 43.59% to 42.31%.

Of 122 restored documents, 119 exceed the strict FP16 score tolerance. This is
not 119 wrong answers, but it rules out a numerical-equivalence claim.

Reproduction command:

```bash
python experiments/run_quality.py matrix configs/int8_accuracy_confirmation.json \
  --profile confirmation --execute --resume
```

## Completed follow-up 3: timing variance

Three paired runs now support the document-FP16 timing claim:

| Workload | Paired speedups | Median (range) |
|---|---|---:|
| Random | 1.189x, 1.252x, 1.250x | 1.250x (1.189--1.252x) |
| Zipf | 2.192x, 2.195x, 2.194x | 2.194x (2.192--2.195x) |

Document-cache run-level mean TTFT has a coefficient of variation below 0.13%
for both workloads. Repetitions 2 and 3 preserve every paired label.

Reproduction command:

```bash
python experiments/run_quality.py matrix configs/timing_repetitions.json \
  --profile confirmation --execute --resume
```

The `smoke` profile contains only ten requests and is a functional check; it
must not be included in the timing aggregate.

## Completed implementation hardening before the next matrix

The pre-arena refactor is complete:

- fixed-block and radix caches use bounded, versioned lazy eviction heaps;
- radix tensor bytes and cached tokens are maintained incrementally rather
  than recomputed by a full tree scan after every insertion;
- all strategies implement one structural `PrefixCache` protocol and are
  constructed through a registry-based factory;
- in-process matrices reuse immutable dataset, tokenizer/config, and token-ID
  state while retaining fresh model weights and empty mutable caches per run;
- inference rows now separate tensor `store_s` and combined `restore_s` from
  policy and prefill time;
- tokenization happens once outside the measured model-forward interval; and
- inference analysis accepts a declarative run suite instead of relying on one
  hard-coded list and fixed plot ranges.

These timing changes introduce result schema `quality-kv-v3`. The frozen 1.5B
archive remains v2 and is not invalidated; it must not be concatenated with new
v3 rows. The new 0.5B scale check is internally aligned and entirely v3.

## Completed follow-up 4: matched-working-set 0.5B confirmation

The six-run Qwen2.5-0.5B scale check is complete at 22.80% of its FP16
article-KV working set. Document LRU reaches `1.206x` cache-only speedup on
random traffic and `2.154x` on Zipf, with 20.31% and 55.93% article-token hit.
Every FP16 cached label agrees with its aligned segmented reference.

Document caching has the lowest random mean TTFT. It is 4.6% faster than the
tuned fixed-block-256 baseline and 0.4% faster than radix, while retaining much
smaller metadata and policy overhead. The result meets all planned exit
criteria and supports scale robustness within the Qwen2.5 family. Protocol,
tables, caveats, and provenance are frozen in
[`qwen_0.5b_results.md`](qwen_0.5b_results.md).

## Remaining implementation: KV arena and Triton restore

The strongest additional systems contribution is a small document-owned arena
plus a fused INT8 restore kernel:

1. Preallocate per-layer K/V slabs and return allocation handles rather than
   independent Python tensor objects.
2. Track a free list, document ownership, generation counters, live bytes,
   metadata bytes, and stranded bytes.
3. Preserve the current cache-policy interface so document, fixed-block, and
   radix organizations can use either the existing tensor backend or the arena.
4. Implement a Triton kernel that reads INT8 K/V and per-layer/per-KV-head
   scales, dequantizes to FP16, and writes directly into the accelerator arena.
5. Time quantization lookup, host-to-device movement, kernel execution, and
   cache reconstruction separately. Schema v3 reports the current combined
   PyTorch path only as `restore_mean_s`; its isolated `dequant_mean_s` and
   `transfer_mean_s` fields remain zero until those stages are instrumented.
6. Microbenchmark PyTorch versus Triton by restored tokens and bytes, then
   rerun only segmented, document FP16, and document INT8 end-to-end paths.

Required checks include exact byte bounds, stale-handle rejection after
eviction, no use-after-free, deterministic allocation traces, identical FP16
labels, recorded INT8 score error, and a CPU/PyTorch fallback when Triton or
CUDA is unavailable.

## Final report sequence

1. Define repeated stable-document QA and distinguish it from open-corpus RAG.
2. Establish capacity, locality, and policy behavior with the complete
   no-inference test trace.
3. Present fair TTFT against segmented execution, with the full one-forward
   path shown only as a separate end-to-end comparison.
4. Show why block 256 is the fair generic baseline and why document ownership
   still reduces management cost.
5. Present INT8 as a memory/latency/accuracy frontier, including both changed
   questions and the restore-timer limitation.
6. Report the three-run timing medians and ranges.
7. Add the second-model check and arena/Triton microbenchmark if completed;
   otherwise list them explicitly as future work.
