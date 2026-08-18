# QuALITY KV Cache

Memory-bounded KV-cache management for repeated question answering over long,
stable documents. This is a controlled document-grounded QA deployment pattern,
not open-corpus RAG: requests repeatedly target one of 381 known QuALITY articles.

QuALITY contexts average roughly 5,159 tokens, while each article has many real
multiple-choice questions. The reusable model prefix is therefore the exact
system prompt plus the complete article. Only the question and four options are
processed on every request.

## Dataset protocol

Use only the official HTML-stripped QuALITY v1.0.1 release. Each article appears
in two writer records; the loader merges records by `article_id`, checks that the
article text agrees, and preserves every question.

| Split | Articles | Questions | Grouped upper-bound hit rate |
|---|---:|---:|---:|
| Train | 150 | 2,523 | 94.1% |
| Dev | 115 | 2,086 | 94.5% |
| Test | 116 | 2,128 | 94.5% |
| Total | 381 | 6,737 | 94.3% |

Test labels are withheld. Use train while developing, dev for final accuracy,
and test for label-free cache and latency traces. Real inference on test can
produce predictions and TTFT measurements, but not accuracy. Download the release from the
[official QuALITY repository](https://github.com/nyu-mll/quality), then validate a
split:

```bash
python -m src.quality_cache.cli validate-data \
  data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.train --split train --verify-counts
```

## Design

The stable system-prompt KV is pinned as L0 and excluded from the cache budget.
Three organizations can be selected with `--cache-strategy`: `document` stores
and evicts one complete article KV object, `fixed-block` uses independently
evicted content-addressed token blocks, and `radix` stores shared token prefixes
in a compressed radix tree. Identity includes model and tokenizer revisions,
prompt version, dtype, and quantization format; document entries additionally
include article ID and content hash. Fixed-block eviction maintains child counts
and policy-ordered leaf heaps, avoiding a full scan of the block table for every
eviction.

Budgets use actual stored tensor bytes and may also cap article count. A budget
that cannot hold one requested article is rejected. Online policies are LRU, LFU
with LRU tie-breaking, FIFO, and GDSF. The simulator also implements offline
farthest-next-use. It is Belady-optimal for equal-size objects but only a
clairvoyant heuristic for variable-size article byte budgets. Matrix storage
modes are accelerator-resident FP16 and CPU symmetric INT8 with separate scales
for each layer and KV head. `accelerator-fp16` resolves to CUDA or MPS from
`--device`, so model weights and cached KV remain on the same accelerator. CPU
restoration, transfer, and INT8 dequantization are part of TTFT.

The model scores the next-token probability of A/B/C/D. If any label is not a
single token, it falls back to full option-sequence likelihood.

## Repository layout

The implementation is grouped by responsibility:

```text
src/quality_cache/
├── caches/       cache types, document/fixed-block/radix strategies, offline policy
├── data/         QuALITY loader, record types, and workload generation
├── inference/    model runner, no-inference runner, and KV tensor storage
├── reporting/    metrics, manifests, serialization, and figures
├── simulation/   inference-free trace simulation
├── cli.py        unified command-line entry point
├── prompt.py     stable prompt construction
└── costs.py      measured prefill-cost model
```

## Install and run

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA runtime on Google Colab, follow the copy-paste setup and smoke-test
commands in [docs/colab.md](docs/colab.md). Dataset files, model weights, and
generated results are downloaded inside Colab and remain excluded from Git. Use
`requirements-colab.txt` there; the fully pinned `requirements.txt` is intended
for a clean virtual environment, not Colab's managed Python installation.

`run` performs real inference by default. Add `--no-inference` to load only the
tokenizer and model configuration, calculate the exact 1.5B KV geometry, and
replay cache decisions without loading weights or calling a model forward pass:

```bash
python experiments/run_quality.py run \
  data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.train \
  --split train --verify-counts \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --tokenizer models/Qwen2.5-1.5B-Instruct \
  --cache-strategy document --policy lru --budget-mb 2048 \
  --storage accelerator-fp16 --device mps --no-inference \
  --workload random --seed 42 --limit 10 \
  --output results/train_random42_document_2gib_no_inference.jsonl
```

Repeat with `--cache-strategy fixed-block` or `radix`. Fixed-block mode uses
`--block-tokens 16` by default. Use `--budget-percent` instead of `--budget-mb`
for a budget relative to the full split working set. No-inference rows report
prefix reuse and simulated prefill work; TTFT and accuracy remain unmeasured.
Use `--limit N` to run only the first or sampled `N` queries; omit it to run all
queries in the selected split/workload. Runs print cache occupancy, eviction
count, and request throughput every 100 requests. Change the interval with
`--progress-every N`, or disable progress with `--progress-every 0`.

Before GDSF no-inference sweeps, calibrate article prefill on the same model and
hardware. Supply the generated piecewise model to `run` or `simulate` with
`--prefill-calibration`:

```bash
python experiments/run_quality.py calibrate-prefill \
  data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.train \
  --split train --verify-counts \
  --model models/Qwen2.5-1.5B-Instruct --device mps \
  --samples 3 --repeats 1 \
  --output configs/prefill_qwen2.5_1.5b_mps.json
```

Omit `--no-inference` to load the weights and perform real inference:

```bash
python experiments/run_quality.py run \
  data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.dev \
  --split dev --verify-counts \
  --model models/Qwen2.5-1.5B-Instruct --device mps \
  --storage accelerator-fp16 --cache-strategy document \
  --policy lru --budget-mb 2048 --workload random --seed 42 \
  --limit 10 --validate-agreement \
  --output results/dev_document_512mb_inference.jsonl
```

Run the requested CUDA version on a GPU host with the model and KV cache both
resident on that GPU:

```bash
python experiments/run_quality.py run \
  data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.test \
  --split test --verify-counts \
  --model models/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype float16 --storage accelerator-fp16 \
  --cache-strategy document --policy lru --budget-mb 4096 \
  --workload random --seed 42 --limit 100 \
  --output results/test_cuda_random42_limit100_document_lru_4gib.jsonl
```

The same command with `--device mps` keeps the cache on MPS. A CUDA run fails
immediately with a clear error when CUDA is unavailable instead of silently
falling back to CPU. The exact CUDA run is also recorded in
`configs/gpu_inference.json`. See [docs/slurm.md](docs/slurm.md) for the complete
cluster copy, environment setup, submission, monitoring, and result-retrieval
workflow.

Run the matched uncached baseline with `--policy none`; it does not require a
cache budget:

```bash
python experiments/run_quality.py run \
  data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.dev --split dev \
  --model models/Qwen2.5-1.5B-Instruct --device cpu \
  --storage cpu-fp16 --policy none \
  --output results/dev_uncached.jsonl
```

The older `simulate` command remains available for multi-policy trace sweeps and
farthest-next-use comparisons. It accepts absolute `--budget-mb` values or
working-set percentages. The matrix uses seed 42 and 4/8 GiB budgets. Fast
checks use ten random requests; confirmation subsets use fifty.

If `--cold-requests` is omitted, the first 10% of each trace is reported as cold
start and the remainder as steady state. The cache-strategy matrix is recorded
in `configs/cache_strategies.json`. It uses only the test split and expands to
108 combinations: three cache strategies, three online policies, two storage
modes, three workloads, and two budgets, all with seed 42.

Validate the matrix without executing it:

```bash
python experiments/run_quality.py matrix configs/cache_strategies.json \
  --profile smoke
```

Run the 10-query smoke matrix, or the complete 2,128-query matrix:

```bash
python experiments/run_quality.py matrix configs/cache_strategies.json \
  --profile smoke --execute --resume
python experiments/run_quality.py matrix configs/cache_strategies.json \
  --profile full --execute --resume
```

The `confirmation` profile uses 50 requests. `--resume` skips result JSONL files
that already exist, and `--max-runs N` can restrict execution while checking the
runner. Test-only profiles always add `--no-inference`; labels, accuracy, TTFT,
and model-answer agreement are therefore unavailable.

Collect run summaries and create the four primary figures:

```bash
python experiments/run_quality.py collect results/*.summary.csv results/simulation.csv \
  --output results/all_summaries.csv
python experiments/run_quality.py plot results/all_summaries.csv figures
```

Every result receives a neighboring manifest containing the result schema and dataset checksum,
seed, exact model/tokenizer identifiers, prompt version, policy, workload,
storage format, budget, git revision, and hardware information. `collect`
rejects legacy or mixed-schema summaries.

The completed 108-run trace findings, representative measurements, and rules
for interpreting partial-hit metrics are recorded in
[`docs/no_inference_results.md`](docs/no_inference_results.md).
The selected eight-run real-inference follow-up and its copy-paste Colab
commands are recorded in
[`docs/inference_confirmation.md`](docs/inference_confirmation.md).

## Measurements and verification

Per-request records contain `cache_hit_ratio = cached_prompt_tokens /
total_prompt_tokens`, so every lookup has a value in `[0, 1]`. The numerator
includes the pinned system-prefix tokens plus the matched article-prefix tokens;
the uncached question/options suffix remains in the denominator. Records also
retain the whole-document boolean, article/tree hit ratios, matched token and
byte counts, and useful/shared/stranded bytes. Document cache is represented as a
one-level tree: the pinned L0 root plus one atomic child per article. The fields
`document_tree_hit_ratio`, `document_tree_cached_tokens`,
`document_tree_total_tokens`, and `partial_document_tree_hit` expose incomplete
tree hits. The compatibility fields `article_cache_hit_ratio` and
`partial_article_hit` use this tree interpretation for every strategy. The
stricter article-text-only values are `article_text_cache_hit_ratio` and
`partial_article_text_hit`, while `root_only_hit` identifies requests that reuse
L0 but no article tokens. These outcomes partition a cached trace into a full
document hit, partial article-text hit, or root-only hit. Radix and fixed-block
nodes store their token counts explicitly, allowing a lookup to accumulate
partial article-prefix hits along its matched path. Cross-strategy tables must
use the explicit tree and article-text columns rather than mixing their
meanings. Summaries report both the per-request macro average and
`article_token_hit_rate = sum(matched article tokens) / sum(requested article
tokens)`, plus avoided
article tokens, cold or
steady phase, TTFT, lookup/load/transfer/dequantization/policy time, prefill time,
occupancy, insertions/evictions, estimated metadata footprint, RSS plus
MPS/CUDA absolute and baseline-delta memory, prediction, gold label, and the
QuALITY-hard flag. Summaries add TTFT mean/p50/p95, accuracy, hard accuracy, and
amortized online/offline prefill cost.

Run the CPU test suite with:

```bash
python -m unittest discover -s tests -v
```

The tests cover merge/count validation, deterministic workloads, known policy
and farthest-next-use traces, all three cache organizations, partial-prefix restoration,
hard byte bounds, pinned L0, INT8 round trips/scales/accounting, proof that
no-inference mode never loads model weights, CPU tensor integration, and a
20,000-request simulation target. The MPS smoke test skips automatically when
MPS is absent.

See [docs/report.md](docs/report.md) for the report structure and limitations.
