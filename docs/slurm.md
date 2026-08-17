# Running the CUDA experiment with Slurm

The prepared job is `slurm/quality_cuda_100.sbatch`. It runs real
Qwen2.5-1.5B inference on the QuALITY test split with random seed 42, 100
queries, a 4 GiB document/LRU cache, and `accelerator-fp16`. Both model weights
and cached KV tensors therefore remain on the allocated CUDA GPU.

## 1. Check the cluster configuration

On the login node, inspect the available partitions, GPU resources, CUDA
modules, and Python modules:

```bash
sinfo -o "%P %a %l %D %G"
module avail cuda
module avail python
```

Edit the `#SBATCH --partition=students` line if the GPU partition has another
name. Some clusters use a typed request such as `--gres=gpu:a4500:1`; use the
syntax reported by `sinfo` or the cluster documentation. The job requires one
GPU with at least 16 GiB, and approximately 20 GiB is preferred. The four-hour
limit is intentionally conservative for long-context inference.

## 2. Copy the project to the cluster

From the local machine, replace the host and destination below. This command
copies the source, HTML-stripped dataset, and downloaded 1.5B model, while
excluding local virtual environments and generated results:

```bash
REMOTE="your_user@login.cluster.example"
REMOTE_DIR="~/rag-kvcache"

ssh "$REMOTE" "mkdir -p $REMOTE_DIR"
rsync -avh --progress \
  --exclude '.git/' \
  --exclude 'venv/' \
  --exclude 'venv-mps/' \
  --exclude '__pycache__/' \
  --exclude 'results/' \
  ./ "$REMOTE:$REMOTE_DIR/"
```

The transfer includes roughly 3 GB of model files. Verify the important files
after connecting:

```bash
ssh "$REMOTE"
cd ~/rag-kvcache
ls -lh models/Qwen2.5-1.5B-Instruct/model.safetensors
ls -lh data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.test
```

## 3. Create the cluster environment once

Use a Python version supported by the project (3.10–3.12). Module names differ
between clusters, so select one shown by `module avail python`:

```bash
module load python/3.11 2>/dev/null || module load python
module load cuda/12.1 2>/dev/null || module load cuda

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Confirm that PyTorch was installed with CUDA support. This verifies the build;
`torch.cuda.is_available()` may still be false on a login node with no assigned
GPU:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

If `torch.version.cuda` is `None`, the environment contains a CPU-only PyTorch
build. Install the CUDA-enabled PyTorch build recommended by the cluster
documentation, then rerun `pip install -r requirements.txt`. The allocated
compute-node check inside the job also verifies actual GPU access before loading
the model.

Validate the copied dataset and the job syntax:

```bash
python experiments/run_quality.py validate-data \
  data/quality-v1.0.1/QuALITY.v1.0.1.htmlstripped.test \
  --split test --verify-counts
bash -n slurm/quality_cuda_100.sbatch
```

## 4. Submit and monitor

Submit from the project root so `SLURM_SUBMIT_DIR` points at the correct files:

```bash
cd ~/rag-kvcache
sbatch slurm/quality_cuda_100.sbatch
squeue -u "$USER"
```

Slurm prints a job ID. Follow its log by substituting that ID:

```bash
tail -f quality_cuda_JOBID.log
scontrol show job JOBID
```

After completion, inspect accounting and the generated summary:

```bash
sacct -j JOBID --format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode
ls -lh results/slurm/
python -m json.tool \
  results/slurm/test_cuda_random42_limit100_document_lru_4gib_JOBID.summary.json
```

The manifest beside each result records `device: cuda`, `cache_device: cuda`,
the GPU hardware, model/tokenizer identity, dataset checksum, cache strategy,
policy, workload, and budget. Test labels are withheld, so TTFT and cache metrics
are available but accuracy remains unavailable.

## 5. Copy results back

Run this from the local project directory:

```bash
rsync -avh --progress \
  "$REMOTE:$REMOTE_DIR/results/slurm/" results/slurm/
scp "$REMOTE:$REMOTE_DIR/quality_cuda_JOBID.log" .
```

To override defaults without editing the script, export variables with Slurm:

```bash
sbatch --export=ALL,MIN_GPU_GIB=20,VENV_DIR="$HOME/rag-kvcache/venv" \
  slurm/quality_cuda_100.sbatch
```
