# CUDA inference confirmation

The complete no-inference matrix identified the comparisons worth confirming
with actual model execution. `configs/inference_confirmation.json` encodes the
resulting eight-run suite on the labelled QuALITY development split and
`Qwen/Qwen2.5-1.5B-Instruct`.

## Selected runs

| # | Workload | Cache | Policy | Storage | Purpose |
|---:|---|---|---|---|---|
| 1 | Random | None | None | accelerator FP16 | Uncached TTFT and accuracy baseline |
| 2 | Random | Document | LRU | accelerator FP16 | Main cached baseline and cached/uncached agreement |
| 3 | Random | Fixed-block | LRU | accelerator FP16 | Organization comparison |
| 4 | Random | Radix | LRU | accelerator FP16 | Organization comparison |
| 5 | Random | Document | LRU | CPU INT8 | Restore/dequantization, agreement, and accuracy delta |
| 6 | Zipf | None | None | accelerator FP16 | Uncached skewed-workload baseline |
| 7 | Zipf | Document | LRU | accelerator FP16 | Policy comparison |
| 8 | Zipf | Document | GDSF | accelerator FP16 | Policy comparison |

All cached runs use a 4 GiB tensor budget and seed 42. The two document-cache
runs with different storage formats enable `--validate-agreement`; this adds an
untimed uncached reference forward for every request. The other cached runs do
not repeat that expensive validation because the unit and CPU integration tests
already exercise the underlying cache and KV reconstruction components. Their
TTFT and accuracy are still measured normally.

The suite intentionally omits grouped order, LFU, 8 GiB accelerator caching,
and most cross-products. The no-inference results already show that grouped
order saturates the reuse ceiling, LFU is not the winner on the selected
traces, and an 8 GiB GPU cache risks exhausting a 15 GiB Colab accelerator once
model weights and active tensors are included.

## Colab commands

First inspect the generated commands and run all eight configurations for ten
queries. Existing outputs are skipped by `--resume`:

```python
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
%env TOKENIZERS_PARALLELISM=false
!python experiments/run_quality.py matrix \
    configs/inference_confirmation.json \
    --profile smoke --show-commands
!python experiments/run_quality.py matrix \
    configs/inference_confirmation.json \
    --profile smoke --execute --resume
```

If the smoke suite succeeds, run the five random configurations at 100
requests. This is the primary inference comparison:

```python
!python experiments/run_quality.py matrix \
    configs/inference_confirmation.json \
    --profile confirmation --execute --resume --max-runs 5
```

Then resume without the cap to skip those five files and execute the three
Zipf configurations:

```python
!python experiments/run_quality.py matrix \
    configs/inference_confirmation.json \
    --profile confirmation --execute --resume
```

The runner releases unused accelerator allocations between selected runs. The
`full` profile means all 2,086 dev questions and should not be launched until
the 100-request results justify that cost.

## Collect and inspect

```python
!python experiments/run_quality.py collect \
    results/inference_confirmation/confirmation/*.summary.csv \
    --output results/inference_confirmation/confirmation/all_summaries.csv
```

Inspect at least these columns: `ttft_mean_s`, `ttft_p50_s`, `ttft_p95_s`,
`prefill_time_s`, `load_mean_s`, `transfer_mean_s`, `dequant_mean_s`,
`article_token_hit_rate`, `accuracy`, `quality_hard_accuracy`,
`reference_label_agreement`, `accuracy_delta_vs_fp16`, and accelerator/RSS
memory peaks. Do not mix these measured inference values with the simulated
prefill times in the no-inference matrix.
