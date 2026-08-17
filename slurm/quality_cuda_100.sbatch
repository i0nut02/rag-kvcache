#!/bin/bash
#SBATCH --job-name=quality_kv_cuda
#SBATCH --output=quality_cuda_%j.log
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
DATASET="$PROJECT_DIR/data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.test"
MODEL_DIR="$PROJECT_DIR/models/Qwen2.5-1.5B-Instruct"
RESULT_DIR="$PROJECT_DIR/results/slurm"
RESULT_PATH="${RESULT_PATH:-$RESULT_DIR/test_cuda_random42_limit100_document_lru_4gib_${SLURM_JOB_ID}.jsonl}"
MIN_GPU_GIB="${MIN_GPU_GIB:-16}"
PYTHON_MODULE="${PYTHON_MODULE:-}"
export MIN_GPU_GIB

finish() {
    status=$?
    echo ">>> Finished at $(date --iso-8601=seconds 2>/dev/null || date) with status $status"
    exit "$status"
}
trap finish EXIT

echo ">>> Job: ${SLURM_JOB_ID:-local} on $(hostname)"
echo ">>> Project: $PROJECT_DIR"
echo ">>> Started: $(date --iso-8601=seconds 2>/dev/null || date)"

cd "$PROJECT_DIR"

if type module >/dev/null 2>&1; then
    if [[ -n "$PYTHON_MODULE" ]]; then
        module load "$PYTHON_MODULE"
    fi
    module load cuda/12.1 2>/dev/null || module load cuda
else
    echo ">>> No module command found; using the CUDA environment already provided."
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: virtual environment not found at $VENV_DIR" >&2
    echo "Create it on the cluster before submitting this job." >&2
    exit 2
fi
source "$VENV_DIR/bin/activate"

# Include CUDA libraries installed inside the Python environment without
# hardcoding a Python minor version. PyTorch wheels normally carry these under
# site-packages/nvidia/*/lib.
PYTHON_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
CUDA_LIBRARY_PATH=""
if [[ -d "$PYTHON_SITE/nvidia" ]]; then
    while IFS= read -r library_dir; do
        CUDA_LIBRARY_PATH="${CUDA_LIBRARY_PATH:+$CUDA_LIBRARY_PATH:}$library_dir"
    done < <(find "$PYTHON_SITE/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -print)
fi
if [[ -n "$CUDA_LIBRARY_PATH" ]]; then
    export LD_LIBRARY_PATH="$CUDA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo ">>> Allocated GPU"
nvidia-smi

python - <<'PY'
import os
import sys
import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: this PyTorch environment cannot access the allocated CUDA GPU")

properties = torch.cuda.get_device_properties(0)
total_gib = properties.total_memory / 2**30
minimum_gib = float(os.environ.get("MIN_GPU_GIB", "16"))
print(f"GPU: {properties.name}")
print(f"GPU memory: {total_gib:.1f} GiB")
if total_gib < minimum_gib:
    raise SystemExit(
        f"ERROR: GPU has {total_gib:.1f} GiB; this job requires at least "
        f"{minimum_gib:.1f} GiB for weights, temporary tensors, and a 4 GiB KV budget"
    )
PY

for required_path in "$DATASET" "$MODEL_DIR/config.json" "$MODEL_DIR/model.safetensors"; do
    if [[ ! -e "$required_path" ]]; then
        echo "ERROR: required file not found: $required_path" >&2
        exit 2
    fi
done

mkdir -p "$RESULT_DIR"

echo ">>> Running real CUDA inference"
echo ">>> Result: $RESULT_PATH"
python experiments/run_quality.py run \
    "$DATASET" \
    --split test \
    --verify-counts \
    --model "$MODEL_DIR" \
    --device cuda \
    --dtype float16 \
    --storage accelerator-fp16 \
    --cache-strategy document \
    --policy lru \
    --budget-mb 4096 \
    --workload random \
    --seed 42 \
    --limit 100 \
    --output "$RESULT_PATH"

echo ">>> Summary"
python - "$RESULT_PATH" <<'PY'
import json
import pathlib
import sys

summary = pathlib.Path(sys.argv[1]).with_suffix(".summary.json")
payload = json.loads(summary.read_text(encoding="utf-8"))
fields = (
    "requests",
    "distinct_articles",
    "full_document_hit_rate",
    "article_token_hit_rate",
    "ttft_mean_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "cache_bytes_peak",
    "cuda_allocated_bytes_peak",
    "cuda_reserved_bytes_peak",
    "insertions",
    "evictions",
    "device",
    "cache_device",
)
print(json.dumps({field: payload.get(field) for field in fields}, indent=2))
PY

nvidia-smi
