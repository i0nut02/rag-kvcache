# Qwen2.5-0.5B matched-working-set confirmation

This is the reproducibility protocol for the completed empirical step after the
frozen Qwen2.5-1.5B baseline. It is a six-run scale check, not a new exhaustive
matrix. Its validated findings are recorded in
[`qwen_0.5b_results.md`](qwen_0.5b_results.md).

| Workload | Runs |
|---|---|
| Random | segmented control, document LRU, fixed-block 256 LRU, radix LRU |
| Zipf | segmented control, document LRU |

All runs use QuALITY dev, seed 42, 100 aligned requests, accelerator FP16, and
the same prompt. Cached runs use `22.80130165664403%` of the smaller model's
FP16 article-KV working set. This is the fraction represented by 4 GiB in the
frozen 1.5B experiment:

```text
4,294,967,296 / 18,836,500,480 = 22.80130165664403%
```

For Qwen2.5-0.5B this is approximately 1.714 GiB, but the runner computes the
exact byte budget from the loaded tokenizer and model geometry and records both
the percentage and working-set bytes in every summary and manifest.

## 1. Prepare Colab

Select a CUDA GPU runtime, then update an existing checkout:

```python
!nvidia-smi
%cd /content/rag-kvcache
!git pull
!python -m pip install --quiet -r requirements-colab.txt
```

For a new runtime, clone first:

```python
!git clone https://github.com/i0nut02/rag-kvcache.git
%cd /content/rag-kvcache
!python -m pip install --quiet -r requirements-colab.txt
```

Use `requirements-colab.txt`, not `requirements.txt`, so Colab's CUDA-enabled
PyTorch installation is not replaced. Set the allocator and tokenizer options
before loading the model:

```python
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
%env TOKENIZERS_PARALLELISM=false
```

Download and validate the labelled development split if it is not already
present:

```python
!mkdir -p data/quality-v1.0.1
!wget -q -P data/quality-v1.0.1 \
  https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev
!python experiments/run_quality.py validate-data \
  data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.dev \
  --split dev --verify-counts
```

The validation must report 115 articles and 2,086 questions.

## 2. Verify code and matrix expansion

Run the CPU suite and inspect, but do not execute, the six generated commands:

```python
!python -m unittest discover -s tests -q
!python experiments/run_quality.py matrix \
  configs/qwen_0.5b_confirmation.json \
  --profile smoke --show-commands
```

The expansion must report six combinations. Cached commands must contain
`--budget-percent 22.80130165664403`; they must not contain a hard-coded 4 GiB
budget.

## 3. Run the functional smoke suite

```python
!python experiments/run_quality.py matrix \
  configs/qwen_0.5b_confirmation.json \
  --profile smoke --execute --resume
```

This runs only ten requests per path. It checks downloads, CUDA execution,
reference alignment, and memory, but its timings are not reportable evidence.
If all six runs finish, inspect their summaries:

```python
import glob, json, os

for path in sorted(glob.glob(
    "results/qwen_0.5b_confirmation/smoke/*.summary.json"
)):
    row = json.load(open(path))
    print(
        os.path.basename(path),
        row["requests"],
        row["budget_percent"],
        row["working_set_bytes"],
        row["reference_label_agreement"],
    )
```

## 4. Run the 100-request confirmation

```python
!python experiments/run_quality.py matrix \
  configs/qwen_0.5b_confirmation.json \
  --profile confirmation --execute --resume
```

The segmented control for each workload is executed first and becomes the
offline JSONL reference for its cached runs. This avoids a second full forward
while the cache occupies GPU memory. `--resume` is safe for interrupted runs,
but only within this new output directory; do not copy old v2 JSONLs into it.

The new code writes result schema `quality-kv-v3`. The frozen 1.5B artifacts
remain `quality-kv-v2` because their timing boundary included different host
work. Never collect v2 and v3 summaries into one table. Compare their
already-aggregated conclusions explicitly instead.

## 5. Analyze the configured suite

```python
!python experiments/run_quality.py analyze-inference \
  results/qwen_0.5b_confirmation/confirmation \
  --suite-config configs/qwen_0.5b_analysis.json \
  --output-dir results/qwen_0.5b_confirmation/analysis
```

The analyzer validates manifests, one result schema, aligned trace positions,
and segmented controls. It writes per-run summaries, cache-only speedups,
correctness comparisons, mismatch details, and three figures. There is no full
one-forward control in this compact suite, so end-to-end speedup is intentionally
`NaN`; use `cache_only_speedup` against segmented inference.

The FP16 exit criteria are:

- all cached runs have `reference_label_agreement == 1.0`;
- trace and manifest validation pass;
- document LRU remains competitive with fixed-block 256 and radix on random;
- report the random and Zipf cache-only speedup and article-token hit rate;
- treat a smoke result as functional evidence only.

Do not run the optional CPU INT8 follow-up until these FP16 results have been
checked. Do not start the arena/Triton matrix yet; that backend is not part of
this confirmation.

## 6. Download the artifacts

```python
from google.colab import files

!zip -qr qwen-0.5b-confirmation.zip results/qwen_0.5b_confirmation
files.download("qwen-0.5b-confirmation.zip")
```
