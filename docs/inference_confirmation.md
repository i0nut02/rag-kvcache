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

All cached runs use a 4 GiB tensor budget and seed 42. Each cached random run
uses run 1 as an offline FP16 reference; each cached Zipf run uses run 6. The
runner joins requests by trace position and request ID, then compares all
A/B/C/D scores and predicted labels without executing a second model forward.
The positional component is necessary because Zipf sampling can cycle back to
the same real question. Label disagreements, maximum logit deltas, and FP16
tolerance violations are recorded rather than aborting a long measurement.
`--strict-reference` remains available for fail-fast smoke tests.

This offline-reference design is important on a 15 GiB T4. Keeping a nearly
full 4 GiB GPU KV cache resident while executing another complete 6k-token
uncached attention pass can exhaust accelerator memory. Reusing the saved
baseline avoids that failure and removes duplicated compute from wall time;
the reference file and its checksum are recorded in every dependent manifest.

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

The runner releases unused accelerator allocations between selected runs. With
`--resume`, a completed baseline is retained and reused by dependent runs. The
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
`reference_label_agreement`, `reference_max_label_logit_delta`,
`reference_tolerance_violations`, `accuracy_delta_vs_fp16`, and accelerator/RSS
memory peaks. Do not mix these measured inference values with the simulated
prefill times in the no-inference matrix.

## Segmented no-document-cache control

The first confirmation exposed an execution-path confound: a document-cache
miss still ran substantially faster than the one-forward uncached baseline.
Because the pinned L0 root is only a small part of the prompt, that difference
cannot be attributed to L0 reuse alone. The fair control therefore executes the
same `L0 -> article -> question/options` path as a document-cache miss but never
inserts or retains article KV between requests.

This is selected with `--policy none --baseline-mode segmented`. Its article
reuse, avoided prefill tokens, cache bytes, insertions, and evictions must all
remain zero. L0 is still recorded as a root-only tree match so that the generic
prefix metrics retain their existing definitions; use `article_token_hit_rate`
when verifying that this control has no document reuse.

After pulling the update in the same Colab checkout, inspect and run the
10-query smoke control:

```python
%cd /content/rag-kvcache
!git pull --ff-only
!python -m unittest discover -s tests -q
!python experiments/run_quality.py matrix \
    configs/segmented_baseline_confirmation.json \
    --profile smoke --show-commands
!python experiments/run_quality.py matrix \
    configs/segmented_baseline_confirmation.json \
    --profile smoke --execute --resume
```

Then execute the 100-query control:

```python
!python experiments/run_quality.py matrix \
    configs/segmented_baseline_confirmation.json \
    --profile confirmation --execute --resume
```

The configuration deliberately uses the same output directory and the same
names for the two full uncached baselines as the original confirmation matrix.
Consequently, `--resume` skips those expensive baselines when they are still
present and runs only:

- `dev_confirmation_01b_segmented_uncached_random_fp16.jsonl`
- `dev_confirmation_06b_segmented_uncached_zipf_fp16.jsonl`

If either full baseline is absent, the matrix regenerates it before its matched
segmented run. Each segmented result uses that full JSONL as an offline
correctness reference. Once both controls finish, recollect all summaries:

```python
!python experiments/run_quality.py collect \
    results/inference_confirmation/confirmation/*.summary.csv \
    --output results/inference_confirmation/confirmation/all_summaries.csv
```

For the report, present two speedups separately: full uncached mean TTFT divided
by cached mean TTFT is the end-to-end system improvement; segmented-control mean
TTFT divided by cached mean TTFT isolates cross-request article-KV reuse. Do not
describe the small pinned L0 root as a separate cache strategy.

## Reproduce the final analysis

After all ten JSONLs and their manifests are in one directory, validate the
archive and generate the report tables, paired-bootstrap intervals, mismatch
diagnostics, and figures with:

```bash
python experiments/run_quality.py analyze-inference \
  results/inference_confirmation/confirmation \
  --output-dir docs/generated/inference_confirmation \
  --bootstrap-samples 20000 --seed 42
```

The analyzer rejects missing runs, mixed models, mixed dataset checksums, mixed
Torch versions, non-aligned traces, duplicate trace positions, and a segmented
control that retained article KV. It records SHA-256 hashes for all input
JSONLs in `analysis.json`.

The checked-in analysis of the completed T4 archive is
[`generated/inference_confirmation/results.md`](generated/inference_confirmation/results.md).
The main cache-only findings are `1.19x` for document/radix FP16 and `1.33x`
for document CPU INT8 on random traffic, and `2.19x` for document FP16 on Zipf
traffic. The 16-token fixed-block implementation is slower than the fair
control (`0.83x`). All FP16 cache paths preserve the segmented-path label;
CPU INT8 changes one of 100 labels in this initial sample.

The targeted follow-ups are now complete. The 256-token fixed-block run, the
300-request INT8 comparison, and three-run timing aggregate are reported in
[`results.md`](results.md). See [`next_steps.md`](next_steps.md) for the
remaining second-model and arena/Triton work.
