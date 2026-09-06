# Running on Google Colab

The commands below run the current QuALITY cache implementation on a Colab
CUDA runtime. Select **Runtime > Change runtime type > T4 GPU** (or a stronger
GPU) before running them. The QuALITY data, model weights, and generated results
are intentionally not stored in Git.

## 1. Clone and install

```python
!nvidia-smi
!git clone https://github.com/i0nut02/rag-kvcache.git
%cd rag-kvcache
!python -m pip install --quiet --upgrade pip
!python -m pip install --quiet -r requirements-colab.txt
```

Do not install `requirements.txt` over Colab's managed environment. That file
pins the complete stack for a clean virtual environment and would replace
Colab's mutually compatible CUDA-enabled Torch, torchvision, and NumPy builds.
The Colab requirements deliberately leave those core packages untouched.

If Colab reports that packages already imported by the runtime were replaced,
restart the runtime once, return to the repository, and continue:

```python
%cd /content/rag-kvcache
```

## 2. Download QuALITY v1.0.1

```python
!mkdir -p data/quality-v1.0.1
!wget -q -P data/quality-v1.0.1 https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.train
!wget -q -P data/quality-v1.0.1 https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev
!wget -q -P data/quality-v1.0.1 https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.test
!python experiments/run_quality.py validate-data data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.dev --split dev --verify-counts
```

## 3. Run the tests and a no-inference smoke run

The no-inference command downloads only the tokenizer and model configuration.
It validates cache behavior without loading the 1.5B model weights.

```python
!python -m unittest discover -s tests -q
!mkdir -p results/colab
!python experiments/run_quality.py run \
    data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.test \
    --split test --verify-counts \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
    --device cuda --dtype float16 --storage accelerator-fp16 \
    --cache-strategy document --policy lru --budget-mb 4096 \
    --workload random --seed 42 --limit 10 --no-inference \
    --output results/colab/test_random42_document_lru_4gib_no_inference.jsonl
```

## 4. Run real inference

Use the labelled development split when measuring accuracy. This first command
uses ten randomly ordered requests as a GPU and memory smoke test:

```python
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
%env TOKENIZERS_PARALLELISM=false
!python experiments/run_quality.py run \
    data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.dev \
    --split dev --verify-counts \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cuda --dtype float16 --storage accelerator-fp16 \
    --cache-strategy document --policy lru --budget-mb 4096 \
    --workload random --seed 42 --limit 10 --validate-agreement \
    --output results/colab/dev_random42_document_lru_4gib_limit10.jsonl
```

After that succeeds, repeat with `--limit 100`. Replace `document` with
`fixed-block` or `radix`, and replace `lru` with `lfu` or `gdsf`, to compare
strategies while keeping the trace fixed.

`--validate-agreement` still requires identical cached and uncached answer
labels, but it executes a second full forward for every request and should only
be used for small standalone checks. The selected confirmation matrix instead
reuses its saved uncached JSONL through `--reference-jsonl`. This provides the
same label/score comparison without duplicated compute or a second attention
workspace. The label-logit check uses a dtype-aware absolute tolerance (0.0625
for FP16); the matrix records violations instead of discarding a long run. Add
`--strict-reference` only when fail-fast behavior is desired.

```python
!python -m json.tool results/colab/dev_random42_document_lru_4gib_limit10.summary.json
```

Do not use the test split for accuracy: its labels are withheld. It is suitable
for cache-hit and latency traces.

To reproduce the initial experiment, use the curated eight-run matrix instead
of manually expanding every strategy and policy. It provides 10-query smoke and
100-query confirmation profiles on QuALITY dev; the exact rationale, run list,
and staged commands are in
[`inference_confirmation.md`](inference_confirmation.md).

After that matrix, run the two segmented no-document-cache controls described
in the same document. They reuse the existing full uncached JSONLs with
`--resume` and isolate article-cache reuse from the effect of executing the
prompt as L0, article, and question segments.

Those runs and the targeted fixed-block, INT8, and timing follow-ups are now
complete. Their results are in [`results.md`](results.md); the corresponding
configs are `fixed_block_inference.json`, `int8_accuracy_confirmation.json`,
and `timing_repetitions.json` under `configs/`. Do not rerun them unless you are
checking reproducibility on a new accelerator.

The six-path Qwen2.5-0.5B confirmation is also complete. Its results are in
[`qwen_0.5b_results.md`](qwen_0.5b_results.md), and its reproducibility commands
remain in [`qwen_0.5b_confirmation.md`](qwen_0.5b_confirmation.md). Do not use a
fixed 4 GiB budget when reproducing it; the experiment matches the 1.5B run's
FP16 working-set fraction.

## 5. Run the complete dev confirmation

The next evidence phase uses all 2,086 labelled dev questions rather than a
100-request sample. Open
[`full_dev_confirmation_colab.ipynb`](../notebooks/full_dev_confirmation_colab.ipynb)
in Colab for a checkpointed workflow. The selected matrix has 12 runs:

- the first six use the random permutation, which visits every dev question
  exactly once and therefore provides standard QuALITY accuracy;
- the final six use the 2,086-request Zipf trace to stress reuse and policy
  behavior; accuracy on this repeated synthetic trace is not standard QuALITY
  accuracy;
- each workload creates one segmented reference JSONL and all of its cached
  runs reuse that reference, avoiding a second full forward and avoidable CUDA
  OOMs;
- fixed-block uses the already tuned 256-token block size, and INT8 uses the
  validated Triton restore implementation.

Validate every path with ten requests first:

```python
!python experiments/run_quality.py matrix configs/full_dev_confirmation.json \
    --profile smoke --execute --resume
```

Then run the full suite in four resumable groups. `--max-runs` selects a prefix
of the matrix, while `--resume` skips JSONLs completed by an earlier group:

```python
!python experiments/run_quality.py matrix configs/full_dev_confirmation.json \
    --profile full --execute --resume --max-runs 3
!python experiments/run_quality.py matrix configs/full_dev_confirmation.json \
    --profile full --execute --resume --max-runs 6
!python experiments/run_quality.py matrix configs/full_dev_confirmation.json \
    --profile full --execute --resume --max-runs 9
!python experiments/run_quality.py matrix configs/full_dev_confirmation.json \
    --profile full --execute --resume
```

On a Colab T4, budget approximately 12--18 GPU-hours in total. The segmented
reference runs dominate elapsed time. The notebook downloads a browser-based
checkpoint after each group, so it does not require Google Drive space and can
be restored into a new runtime. A single interrupted run must restart, but all
previously completed runs remain resumable.

After all 12 JSONLs and manifests are present, generate the validated paired
analysis:

```python
!python experiments/run_quality.py analyze-inference \
    results/full_dev_confirmation/full \
    --suite-config configs/full_dev_analysis.json \
    --output-dir results/full_dev_confirmation/analysis \
    --bootstrap-samples 20000 --seed 42
```

Keep the frozen 100- and 300-request results unchanged until this complete
suite passes analysis. The longer run is an additional evidence layer, not a
silent replacement for earlier artifacts.

## Optional: repeat only the shortlisted strategy timings (ten-hour cap)

Use [strategy_repetitions_colab.ipynb](../notebooks/strategy_repetitions_colab.ipynb)
for the bounded follow-up to the simulation-selected comparisons. It schedules
1,000 queries × 3 repetitions × document/fixed-block-256/radix × random/Zipf,
plus two single correctness references: 20,000 requests, approximately 7–9
hours, with a persistent ten-hour benchmark deadline. This is not a rerun of
the no-inference matrix. No GPU measurements from this new protocol exist yet.

Run its setup, ten-request smoke, then the six numbered groups. The notebook
uses new processes, validates resume receipts, downloads checkpoints without
Drive and produces run-level mean/**p90** comparisons. A bare `matrix --execute`
does not apply its wall-clock cap. See [the exact protocol and timeout
behavior](strategy_repetitions.md), including partial-group exclusion and the
need to disconnect the Colab runtime yourself after the benchmark ends.

## 6. Run the arena and Triton phase

The optional arena and corrected Triton comparison is complete; reproduce it
only if you need a fresh machine-level replication. Pull the current commit,
set allocator configuration before importing Torch, and verify Triton:

```python
%cd /content/rag-kvcache
!git pull
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
%env TOKENIZERS_PARALLELISM=false
!python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"
!python -m unittest discover -s tests -q
```

First run the corrected restore microbenchmark with complete hardware
provenance:

```python
!python experiments/run_quality.py benchmark-restore \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cuda --dtype float16 \
    --tokens 512 2048 8192 \
    --backends pytorch triton --warmup 2 --repeats 10 --seed 42 \
    --output results/arena_triton/restore_microbenchmark_runtime_stride.csv
```

Then run the six aligned 100-request paths: segmented control, ordinary tensor
FP16, arena FP16 with 64- and 256-token pages, and CPU INT8 restored with
PyTorch and Triton. The matrix creates the segmented reference before the
cached runs; `--resume` safely skips a completed JSONL.

```python
!python experiments/run_quality.py matrix configs/arena_triton_confirmation.json \
    --profile confirmation --execute --resume
```

Analyze trace alignment, allocator invariants, correctness, paired speedups,
and startup amortization:

```python
!python experiments/run_quality.py analyze-inference \
    results/arena_triton/confirmation \
    --suite-config configs/arena_triton_analysis.json \
    --output-dir results/arena_triton/analysis \
    --bootstrap-samples 20000 --seed 42
```

The final reference result uses six 100-request rows, not the 20-request smoke
profile. It records a 2.671-second one-time Triton warm-up separately from
online TTFT. PyTorch and Triton INT8 must agree on every A/B/C/D score; both
should have the same one label mismatch against FP16. Exact reference values,
hashes, figures, and interpretation are in
[`generated/arena_triton`](generated/arena_triton/README.md). Do not use smoke
timings as evidence and do not merge schema-v2 rows into this schema-v3 suite.

## 7. Preserve arena results before the Colab runtime expires

```python
from google.colab import files
!du -sh results/arena_triton
!zip -qr arena-triton-results.zip results/arena_triton
files.download("arena-triton-results.zip")
```

This archives only the final phase rather than every earlier matrix, which
keeps Colab disk usage and the download small. After the browser download has
completed and you have verified the ZIP locally, it is safe to remove that ZIP
from the ephemeral Colab runtime. Raw result archives remain intentionally
excluded from Git.
