# Repository guide for coding agents

## Scope

These instructions apply to the entire repository. Read this file before making
changes, then read the relevant source, tests, and documentation for the task.
Do not treat this file as a substitute for inspecting the implementation.

## Project objective

This project studies memory-bounded reuse of transformer KV state for repeated
question answering over long, stable documents. It is a controlled
document-grounded QA workload, not open-corpus RAG:

- every request already contains its target QuALITY `article_id`;
- there is no retrieval or document search stage;
- the reusable prefix is the exact system prompt followed by the complete
  article;
- the question and its four answer options are request-specific suffixes;
- answers are selected by scoring A/B/C/D, with sequence-likelihood fallback
  when labels are not single tokens.

Keep the model-level research question visible. Cache policies alone are not the
contribution; the project connects document boundaries to causal-transformer KV
reuse, memory capacity, TTFT, quantization error, and answer correctness.

## Ground-truth protocol

- Use only the official HTML-stripped QuALITY v1.0.1 files.
- Merge the two writer records for each `article_id`, require identical article
  text, and preserve every question.
- Official totals are 381 articles and 6,737 questions:
  train 150/2,523, dev 115/2,086, and test 116/2,128.
- Use train for development, dev for labeled inference and accuracy, and test
  only for label-free cache traces because test labels are withheld.
- Synthetic workloads use seed 42. `grouped` preserves article locality,
  `random` shuffles real requests, and `zipf` samples articles with exponent
  1.1 while cycling through their real questions.
- The request supplies the exact article. Do not add a retriever unless the
  project scope is explicitly changed.

## Current experiment baseline

The Qwen2.5-1.5B empirical baseline is frozen. Before changing experimental
claims, read:

- `docs/results.md` for consolidated findings and limitations;
- `docs/inference_confirmation.md` for fair-baseline execution;
- `docs/no_inference_results.md` for the complete trace matrix.

Current supported choices are:

- document + LRU + accelerator FP16 is the primary implementation;
- fixed-block with 256-token blocks is the tuned generic/vLLM-like baseline;
- radix is an architectural comparison and is not expected to win for one
  unrelated article per prompt;
- GDSF is workload-dependent and should not be described as a universal win;
- CPU INT8 is a memory/latency/quality tradeoff, not a lossless mode.

Do not report smoke runs as timing evidence. Do not replace the frozen 1.5B
numbers with a single new run. The matched-working-set Qwen2.5-0.5B
confirmation, KV-arena comparison, and corrected prewarmed Triton INT8 restore
comparison are complete. The first length-specialized Triton row is diagnostic
only; final claims use the runtime-stride evidence in `docs/arena_triton.md`
and `docs/generated/arena_triton/` and report JIT warm-up separately.

## Repository map

```text
src/quality_cache/
  caches/       cache interfaces, policies, and document/fixed-block/radix stores
  data/         QuALITY loading, record types, and workload generation
  inference/    model runner, no-inference runner, references, tensor storage
  reporting/    summaries, manifests, analysis, and plotting
  simulation/   inference-free trace simulation
  cli.py        unified CLI
  matrix.py     configuration-matrix expansion and execution
  prompt.py     stable prompt construction and prompt version
  schema.py     result and calibration schema versions
experiments/run_quality.py  thin executable entry point
configs/                   reproducible experiment definitions
tests/                     unit and CPU/accelerator integration tests
docs/                      protocols, results, limitations, and runbooks
slurm/                     CUDA cluster job definitions
```

Keep reusable logic under `src/quality_cache`; keep the experiment entry point
thin. Put strategy-specific code in its own module under `caches/` rather than
growing a monolithic file.

## Cache and metric invariants

- L0 is the stable system-prompt KV, not the query. It remains pinned and is not
  evidence of article reuse. The query/options suffix is never cached.
- Article reuse is the primary signal. Prefer
  `article_token_hit_rate = sum(matched article tokens) / sum(requested article
  tokens)` for cross-strategy comparisons.
- A document cache stores and evicts a complete article as one logical unit.
  It can reuse L0 plus either zero or all article text; it has no partial
  article-text hit by design.
- Fixed-block and radix stores may produce partial article-text hits. Nodes or
  blocks must retain token counts so the matched fraction is computed from
  tokens, not node counts.
- Do not confuse `partial_prefix_hit_rate`, which includes root-only matches,
  with `partial_article_text_hit_rate`.
- Stored tensor bytes must never exceed the configured budget after an
  operation. Account for quantization scales and separately report metadata.
- Evicted objects must become unreachable. Pinned L0 must survive article
  eviction.
- Cache identity must include article ID/content hash where applicable, model
  and tokenizer revisions, prompt version, dtype, and storage/quantization
  format.
- CPU INT8 uses symmetric quantization with per-layer, per-KV-head scales and
  dequantizes to the model dtype during restore.
- Lookup, load, transfer, dequantization, policy time, and real prefill time
  must retain distinct meanings. Never present a combined timer as an isolated
  kernel or transfer measurement.

If a metric name or meaning changes, update its producer, aggregator, docs, and
tests together. Bump `RESULT_SCHEMA_VERSION` for an incompatible result
semantic change; never silently combine legacy and current rows.

## Inference and no-inference modes

`run` performs real model inference unless `--no-inference` is supplied.

- Real inference may report observed TTFT, logits, predictions, and dev
  accuracy.
- No-inference loads tokenizer/config geometry without model weights or model
  forward calls. It may report cache behavior, byte occupancy, token reuse,
  evictions, and simulated prefill cost.
- No-inference TTFT, accuracy, transfer time, and dequantization time are not
  measurements and must not be used as such.
- Use `--limit 10` only for functional smoke tests. Confirmation runs normally
  use 100 aligned dev requests; the established INT8 accuracy run uses 300.
- Use a segmented `--policy none` baseline for cache-only comparisons. A full
  one-forward baseline has a different execution shape and is only an
  end-to-end comparison.
- For accelerator runs, generate an uncached reference JSONL once and pass it
  with `--reference-jsonl`. This avoids keeping a large cache while executing a
  second full forward and prevents avoidable CUDA OOMs.
- Reference rows are matched by trace position and request ID. Request IDs are
  not unique in Zipf traces, so never index a reference by request ID alone.
- Use `--strict-reference` only for small debugging checks. Report tolerance
  violations separately from label mismatches.

## Development commands

Use Python 3.10--3.12. The clean local environment is pinned by
`requirements.txt`; Colab must use `requirements-colab.txt` so its compatible
CUDA PyTorch stack is not replaced.

Run the complete CPU suite before handing off a code change:

```bash
python -m unittest discover -s tests -v
```

When iterating, run the narrow affected test module first, for example:

```bash
python -m unittest tests.test_cache_strategies -v
python -m unittest tests.test_cpu_integration -v
```

Validate official data counts when changing the loader or workloads:

```bash
python experiments/run_quality.py validate-data \
  data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.train \
  --split train --verify-counts
```

Validate matrix expansion without executing inference:

```bash
python experiments/run_quality.py matrix configs/cache_strategies.json \
  --profile smoke
```

See `README.md` and the committed notebooks for full commands. Do not launch a
costly full matrix merely to validate a local change.

## Change-specific verification

- Cache/policy change: deterministic known traces, hard byte bound, eviction
  reachability, L0 pinning, and the 20,000-request no-inference performance
  test.
- Prompt/tokenization change: prompt tests, dataset trace alignment, cache-key
  versioning, and FP16 cached-versus-segmented label agreement.
- Tensor/INT8 change: round-trip shapes, scales, byte accounting, storage
  reduction, logits/labels, and CPU fallback.
- CUDA/MPS change: keep model weights and accelerator-FP16 cache on the same
  selected device; fail clearly when a requested accelerator is unavailable.
- Reference/analysis change: duplicate Zipf request IDs, checksums, contiguous
  indexes, manifest compatibility, and mixed-schema rejection.
- Config/SLURM change: validate profile counts and run the corresponding config
  and job-file tests.
- Arena/Triton change: also test deterministic allocation, stale-handle
  rejection, no use-after-free, fragmentation/stranded bytes, and a
  PyTorch/CPU fallback when Triton or CUDA is unavailable.

## Artifact and documentation discipline

- `data/`, `models/`, `results/`, `figures/`, and logs are intentionally
  ignored. Do not commit downloaded datasets, weights, or raw result archives.
- Every result must have a neighboring manifest containing dataset checksum,
  seed, model/tokenizer revisions, prompt version, strategy, policy, storage,
  budget, code revision, and hardware details.
- Never reuse older results after changing cache accounting, prompts, trace
  construction, or result semantics. Preserve existing raw artifacts unless
  the user explicitly asks to remove them.
- Generated report artifacts under `docs/generated/` may be tracked when they
  are the curated evidence referenced by the report.
- Keep `README.md`, relevant files under `docs/`, configs, and tests synchronized
  with user-visible behavior.
- State limitations explicitly: QuALITY is not retrieval, test accuracy is
  unavailable, calibrated prefill is not observed TTFT, and current evidence is
  not a multi-model or multi-GPU generalization claim.

## Editing and Git expectations

- Preserve unrelated user changes in a dirty worktree.
- Prefer small, scoped changes with tests over broad rewrites.
- Do not hardcode local model, dataset, Colab, or cluster paths.
- Avoid new dependencies unless they are necessary; update the appropriate
  clean-environment and Colab dependency files deliberately.
- Do not commit or push unless the user explicitly asks. Before either action,
  show the changed files and verification result.
