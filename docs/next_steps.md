# Next experimental steps

## What the current evidence says

The fair 100-request confirmation now compares every cache with segmented
inference that follows the same `L0 -> article -> suffix` execution path but
retains no article KV. This removes the large one-forward-versus-segmented
execution confound.

- Document LRU and radix LRU both provide about `1.19x` cache-only speedup on
  random traffic. Radix does not improve latency or hit rate for this
  one-article-per-request workload.
- Document LRU provides about `2.19x` cache-only speedup on Zipf traffic. LRU
  and GDSF are indistinguishable at this sample size.
- The 16-token fixed-block cache is slower than the fair control (`0.83x`),
  despite approximately the same article-token hit rate as document caching.
  Its 21,169 evictions and Python-level block-management work are the likely
  cause.
- CPU INT8 is promising for capacity and latency (`1.33x` on random traffic),
  but it changed one answer out of 100 relative to the segmented FP16 path.
  It must not yet be called lossless.

The reproducible tables, figures, input hashes, and mismatch records are in
[`generated/inference_confirmation`](generated/inference_confirmation/results.md).

## 1. Resolve fixed-block granularity first

Run the cheap, label-free 16/64/256-token sweep on the QuALITY test trace. It
uses only random and Zipf, LRU, accelerator FP16, 4 GiB, and seed 42, so it adds
six targeted runs rather than another cross-product.

```bash
python experiments/run_quality.py matrix configs/fixed_block_sensitivity.json \
  --profile smoke --show-commands
python experiments/run_quality.py matrix configs/fixed_block_sensitivity.json \
  --profile smoke --execute --resume
python experiments/run_quality.py matrix configs/fixed_block_sensitivity.json \
  --profile full --execute --resume
python experiments/run_quality.py collect \
  results/fixed_block_sensitivity/full/*.summary.csv \
  --output results/fixed_block_sensitivity/full/all_summaries.csv
```

Compare `article_token_hit_rate`, `policy_mean_s`, `lookup_mean_s`, evictions,
metadata footprint, and cache footprint. Select a larger block only if it
materially reduces management cost without sacrificing useful token hits. Then
repeat the 100-request inference comparison for only the winning block size.

## 2. Confirm the INT8 quality effect

The next correctness run directly compares CPU INT8 document caching with the
segmented FP16 path over 300 random requests. It avoids the expensive full
one-forward baseline and records offline per-position score comparisons.

```bash
python experiments/run_quality.py matrix configs/int8_accuracy_confirmation.json \
  --profile smoke --execute --resume
python experiments/run_quality.py matrix configs/int8_accuracy_confirmation.json \
  --profile confirmation --execute --resume
```

Report the mismatch count and rate, accuracy delta, QuALITY-hard accuracy
delta, maximum score delta, and the margins of every changed prediction. If
the mismatch rate remains nonzero, present INT8 as a memory/latency/quality
tradeoff rather than imposing an arbitrary claim of numerical equivalence.

## 3. Measure run-to-run timing variance

The paired bootstrap intervals in the current figures capture request
variability, not GPU run-to-run noise. Two additional repetitions use the same
seed-42 traces for the fair segmented and document-FP16 paths on random and
Zipf traffic:

```bash
python experiments/run_quality.py matrix configs/timing_repetitions.json \
  --profile smoke --execute --resume
python experiments/run_quality.py matrix configs/timing_repetitions.json \
  --profile confirmation --execute --resume
```

Combine these with the existing run as repetitions 1--3. Report the median and
range of the three run-level mean TTFT values as well as the request-level
distribution. Do not change the workload seed: these repetitions isolate
system timing noise rather than trace composition.

## 4. Add one scale-confirmation model

After the block-size and INT8 decisions are fixed, repeat only the decisive
comparisons with `Qwen/Qwen2.5-0.5B-Instruct`: segmented control, document LRU,
the winning fixed-block size, and radix on random traffic; segmented plus
document LRU on Zipf traffic. First calculate the model's FP16 corpus working
set and choose a budget with the same working-set fraction as the 1.5B 4 GiB
run. Reusing 4 GiB blindly would give the smaller model an easier cache regime.

This second model is a scale/generalization check, not another complete matrix.
Keep Qwen2.5-1.5B-Instruct as the primary model because all current measured
results and correctness diagnostics use it.

## 5. Implement the systems contribution

Once the empirical baseline is frozen, the strongest implementation extension
is a small KV arena plus a fused Triton INT8 restore kernel:

1. Add an arena backend with preallocated layer slabs, allocation handles, a
   free list, document ownership, generation counters, and exact live/stranded
   byte accounting.
2. Preserve the current cache-policy interface so document, fixed-block, and
   radix organizations can allocate through either the legacy tensor backend
   or the arena.
3. Implement a Triton kernel that reads INT8 K/V and per-layer/per-KV-head
   scales, dequantizes to FP16, and writes directly into the destination arena.
4. Microbenchmark PyTorch restore versus Triton restore by tokens and bytes,
   then rerun the selected end-to-end inference suite.
5. Require identical FP16 labels, bounded INT8 error, no use-after-eviction,
   exact capacity enforcement, and an explicit fallback when Triton/CUDA is
   unavailable.

This is a better course-project contribution than reimplementing the entire
attention stack: it has a clear systems hypothesis, a contained kernel, and
both microbenchmark and end-to-end evaluation paths.

## Final report order

1. Explain the repeated stable-document deployment setting and why it is not
   open-corpus RAG.
2. Establish cache behavior with the complete no-inference test trace.
3. Present fair real-inference TTFT using the segmented no-document-cache
   control; show the one-forward baseline only as an end-to-end system number.
4. Compare document, fixed-block, radix, LRU/GDSF, and FP16/INT8 using the
   smallest set of experiments that answers each research question.
5. Present arena/kernel microbenchmarks, then end-to-end effects and
   limitations.
