# Report blueprint

## Scope and research questions

QuALITY is document-grounded multiple-choice QA rather than open-corpus RAG.
This project studies a controlled deployment pattern: many requests arrive for a
bounded, stable long-document collection. HotpotQA is mentioned only as
motivation for rejecting low-reuse workloads; its data and old measurements are
not part of the evaluation.

The report answers four questions:

1. Does exact article-prefix reuse lower time to first token?
2. Under which request orderings and budgets do eviction policies differ?
3. Does INT8 increase useful capacity without material label or accuracy loss?
4. When is offline precomputation amortized by repeated questions?
5. When does document-aware atomic caching outperform generic fixed-block or
   radix-prefix caching?

## Experimental protocol

Use only the QuALITY v1.0.1 HTML-stripped test file for the configured matrix.
Merge the two raw writer records for each article and retain all 2,128 questions.
Because test labels are withheld, treat these runs strictly as label-free cache
traces and do not report accuracy or QuALITY-hard accuracy from them.

Report grouped, seeded random, and Zipf(1.1) workloads. Use seed 42 for
synthetic traces. Separate cold start from steady
state. Compare document, 16-token fixed-block, and radix caches with LRU, LFU,
and GDSF at 2, 4, and 8 GiB. Fast checks use 10 requests, confirmation runs use
50, and the full profile uses every test question.
Repeat fixed-block sensitivity at 64 and 256 tokens.

All configured runs use the local Qwen2.5-1.5B-Instruct tokenizer and KV geometry
in no-inference mode. Compare accelerator-resident FP16 (CUDA or MPS) and
per-layer/per-KV-head symmetric CPU INT8 accounting. GDSF uses the measured
prefill-cost calibration.

## Primary results

Include exactly these primary figures:

1. fractional request hit (`cached prompt tokens / total prompt tokens`),
   token-weighted hit rate, and byte hit rate versus working-set budget;
2. TTFT p50/p95 versus actual memory budget;
3. policy comparison faceted by workload;
4. FP16/INT8 useful-capacity and cache-footprint tradeoff.

Tables should include avoided prefill tokens, lookup/load/transfer/dequantization
and policy overhead, occupancy/evictions, RSS and MPS/CUDA memory. TTFT, QuALITY
accuracy, QuALITY-hard accuracy, and label agreement require inference and
released labels, so they are outside this test-only matrix.

## Validity and limitations

The document strategy uses one whole-article storage and eviction unit. The
fixed-block strategy admits and evicts completed blocks independently; the radix
strategy restores the longest matching token prefix and evicts cold leaves. L0
is pinned and excluded from capacity for all three. Farthest-next-use is
simulation-only and is not labeled an upper bound for variable-size articles. Approximate trace
tokenization is suitable for relative policy sweeps; actual experiments use the
model tokenizer and measured tensor bytes. CPU transfer and INT8 dequantization
must remain inside TTFT. Results apply to repeated stable-document QA, not to
arbitrary retrieved-document composition or rapidly changing corpora.
