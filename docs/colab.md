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

## 5. Run the arena and Triton phase

The optional arena and Triton backends are now implemented. Pull the current
commit, set allocator configuration before importing Torch, and verify Triton:

```python
%cd /content/rag-kvcache
!git pull
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
%env TOKENIZERS_PARALLELISM=false
!python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name())"
!python -m unittest discover -s tests -q
```

The supplied 100-request archive validates the segmented, tensor FP16,
arena-64, arena-256, and PyTorch INT8 paths. Its Triton row exposed repeated
JIT specialization by article length. After pulling the runtime-stride fix,
preserve that row and rerun only the short benchmark and Triton path. The
complete explanation is in [`arena_triton.md`](arena_triton.md).

First archive the diagnostic and rerun the microbenchmark with complete
hardware provenance:

```python
!mkdir -p results/arena_triton/diagnostic_length_specialized
!cp results/arena_triton/confirmation/dev_confirmation_document_triton_random_int8_4gib* \
    results/arena_triton/diagnostic_length_specialized/
!python experiments/run_quality.py benchmark-restore \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cuda --dtype float16 \
    --tokens 512 2048 8192 \
    --backends pytorch triton --warmup 2 --repeats 10 --seed 42 \
    --output results/arena_triton/restore_microbenchmark_runtime_stride.csv
```

Then rerun Triton in a fresh process. Kernel compilation is moved before the
request loop and reported as `offline_restore_warmup_s` rather than TTFT:

```python
!python experiments/run_quality.py run \
    data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.dev \
    --split dev --verify-counts \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --device cuda --dtype float16 \
    --cache-strategy document --policy lru \
    --storage cpu-int8 --int8-restore-backend triton \
    --budget-mb 4096 --workload random --seed 42 \
    --block-tokens 256 --limit 100 --progress-every 1 \
    --reference-jsonl results/arena_triton/confirmation/dev_confirmation_segmented_random_fp16.jsonl \
    --output results/arena_triton/confirmation/dev_confirmation_document_triton_random_int8_4gib.jsonl
```

Regenerate the validated comparisons:

```python
!python experiments/run_quality.py analyze-inference \
    results/arena_triton/confirmation \
    --suite-config configs/arena_triton_analysis.json \
    --output-dir results/arena_triton/analysis \
    --bootstrap-samples 20000 --seed 42
```

For a fresh reproduction without the supplied archive, the original 20-request
smoke and 100-request matrix remain available through
`configs/arena_triton_confirmation.json`. Do not use smoke timings as evidence.

## 6. Preserve results before the Colab runtime expires

```python
from google.colab import files
!zip -qr quality-colab-results.zip results
files.download("quality-colab-results.zip")
```

The downloaded archive should now include `results/arena_triton` as well as any
previous confirmations. Raw result archives remain intentionally excluded from
Git.
