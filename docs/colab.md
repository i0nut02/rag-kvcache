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
for FP16).

```python
!python -m json.tool results/colab/dev_random42_document_lru_4gib_limit10.summary.json
```

Do not use the test split for accuracy: its labels are withheld. It is suitable
for cache-hit and latency traces.

For the next experiment, use the curated eight-run matrix instead of manually
expanding every strategy and policy. It provides 10-query smoke and 100-query
confirmation profiles on QuALITY dev; the exact rationale, run list, and staged
commands are in
[`inference_confirmation.md`](inference_confirmation.md).

## 5. Preserve results before the Colab runtime expires

```python
from google.colab import files
!zip -qr quality-colab-results.zip results/colab
files.download("quality-colab-results.zip")
```

The current repository contains the legacy tensor-store implementation and the
document, fixed-block, and radix logical cache strategies. The planned paged
arena and Triton restore backend will use additional CLI flags once implemented;
the commands above do not claim those backends are already available.
