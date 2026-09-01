# Document KV arena and Triton restore

This phase adds two optional physical backends without changing the frozen
Qwen2.5-1.5B or Qwen2.5-0.5B claims. The 100-request arena confirmation is
complete. A first matched Triton run exposed token-length-dependent JIT
specialization in the kernel and is retained as diagnostic evidence, not as a
final Triton TTFT result. The fix makes the stride runtime-valued, explicitly
prewarms the one model-geometry kernel outside request timing, and requires
only the Triton row to be rerun.

## Relationship to SGLang

The design follows the separation visible in SGLang rather than copying its
serving stack:

- SGLang explicitly separates request-to-token mappings, allocation of token
  locations, and the physical KV pool in
  [`memory_pool.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py).
- Its
  [`PagedTokenToKVPoolAllocator`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/allocator/paged.py)
  returns page-aligned locations and releases pages through a distinct
  allocator API.
- Its
  [`RadixCache`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
  owns references to those locations and returns them to the allocator when a
  tree leaf is evicted.
- Device kernels write K/V rows into the pool at validated locations; see
  SGLang's
  [`store_cache`](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/kvcache/kvcache.py).

This repository has no concurrent scheduler or retrieval stage, and each
request already supplies exactly one article. Its corresponding layers are:

```text
DocumentPrefixCache     logical ownership and LRU/LFU/FIFO/GDSF eviction
PageAllocator           page ownership, generations, allocation/release
ArenaHandle             immutable document allocation capability
KVArena slabs           preallocated per-layer K/V device tensors
```

A document remains one logical entry even when its physical allocation spans
many pages. This is the main difference from a generic token-prefix server:
eviction and admission use document boundaries known by the workload.

## Arena backend

Select the arena with:

```text
--cache-strategy document
--storage accelerator-fp16
--kv-backend arena
--arena-page-tokens 64|256
```

The arena:

- preallocates FP16 K/V slabs on the selected CUDA or MPS device;
- allocates the lowest available page IDs deterministically;
- stores one generation-checked handle per complete document;
- evicts policy victims before allocating an incoming document;
- rejects stale and foreign handles after release;
- reports reserved, live, useful, free, and stranded bytes separately; and
- keeps L0 outside the arena and pinned across document eviction.

`cache_bytes` is live page allocation, while `arena_reserved_bytes` is the
physical slab reservation. `stranded_bytes` is internal tail-page waste. Tensor
bytes remain bounded by the configured cache budget; metadata is reported
separately, as in the existing backends. Byte-hit rate uses useful matched KV
bytes and excludes page padding, so allocator fragmentation cannot inflate the
ratio above one.

This is an allocator experiment on top of the standard Transformers attention
path, not a page-table-aware attention engine. On a hit, arena pages are still
assembled into a transient contiguous legacy cache before the model forward.
SGLang can consume indexed KV locations inside its serving runtime; matching
that behavior would require replacing the model attention path as well as the
allocator. The present comparison therefore tests deterministic pooled
ownership, fragmentation, and allocation overhead without assuming that arena
storage alone must improve TTFT.

The arena currently supports only the atomic document strategy. Fixed-block
and radix nodes require independently sliceable allocations; adding them before
there is evidence they help this workload would obscure the document-level
experiment and introduce shared-page lifetime rules that are not needed here.

## Triton INT8 restore

CPU INT8 entries retain symmetric per-layer, per-KV-head scales. Restoration
now has three choices:

```text
--int8-restore-backend pytorch
--int8-restore-backend auto
--int8-restore-backend triton
```

The instrumented PyTorch path and the Triton path both transfer INT8 values and
FP32 scales to the selected accelerator before dequantization. The Triton
kernel fuses multiplication by the per-head scale and the cast into the final
FP16/BF16 output; it does not create a full-size FP32 tensor. `auto` selects
Triton only on CUDA when it imports successfully, otherwise it falls back to
PyTorch. Explicit `triton` fails early rather than silently changing the
experiment.

The Triton destination is a model-ready transient device tensor, not the
persistent FP16 document arena. This is intentional: retaining every CPU INT8
entry again in a persistent FP16 arena would duplicate the cache and destroy
the approximately two-times capacity advantage under study. The arena and
Triton runs therefore answer two orthogonal questions:

1. Does pooled device allocation improve accelerator-FP16 document caching?
2. Does fused device-side restoration improve the CPU-INT8 tradeoff?

Per-request timing now records:

- `transfer_s`: source tensors and scales moved to the target device;
- `dequant_s`: PyTorch or Triton dequantization and final cast;
- `load_s`: final block/layer assembly;
- `restore_s`: the outer combined restore wall time.

The component timers are synchronized measurements, but Python dispatch means
they need not sum exactly to `restore_s`.

## Restore microbenchmark result

The pre-fix diagnostic Qwen2.5-1.5B synthetic-geometry run used model revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, 28 layers, two KV heads,
head dimension 128, FP16 output, seed 42, two warm-ups, and ten measured
repetitions. Both requested backends resolved to themselves. The benchmark
uses schema `quality-int8-restore-benchmark-v1` and result schema
`quality-kv-v3`. It was produced at Git revision
`ea3f6fd3cb9b2102810a296eb89c528455411bf4`.

| Tokens | PyTorch restore | Triton restore | Restore speedup | PyTorch throughput | Triton throughput | Triton p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 6.230 ms | 5.427 ms | 1.148x | 1.097 GiB/s | 1.260 GiB/s | 5.550 ms |
| 2,048 | 14.468 ms | 12.412 ms | 1.166x | 1.890 GiB/s | 2.203 GiB/s | 13.332 ms |
| 8,192 | 45.161 ms | 39.854 ms | 1.133x | 2.422 GiB/s | 2.744 GiB/s | 44.039 ms |

The mean restore reductions are 12.88%, 14.21%, and 11.75%. The isolated
dequantization component still scales more strongly:

| Tokens | PyTorch dequant | Triton dequant | Dequant speedup | Triton transfer | Transfer share of Triton restore |
|---:|---:|---:|---:|---:|---:|
| 512 | 2.299 ms | 1.501 ms | 1.532x | 3.824 ms | 70.46% |
| 2,048 | 3.396 ms | 1.665 ms | 2.039x | 10.621 ms | 85.57% |
| 8,192 | 11.233 ms | 3.006 ms | 3.737x | 36.624 ms | 91.89% |

`max_abs_error_vs_pytorch` is exactly zero in all six rows. Peak CUDA
allocation is also slightly lower for Triton: 49.03 versus 49.78 MiB at 512
tokens, 196.03 versus 199.03 MiB at 2,048, and 784.03 versus 796.03 MiB at
8,192.

This supports a narrow claim: the fused Triton dequantization/cast is correct
for these inputs and improves isolated restore throughput. It does not show an
end-to-end TTFT improvement. At 8,192 tokens the measured transfer is 91.89%
of Triton restore time, so host-to-device movement is now the dominant
bottleneck. Pinned/asynchronous transfer and overlap are possible future work,
but require stream-aware lifetime tests and a separate experiment.

The CSV hash is
`3c9982340ed41a82368ce6fe94437f492278469486d7ae9528215ec6d846dee0`.
Its manifest records Linux, Torch `2.11.0+cu128`, and Python `3.13.15`, but the
old manifest collector did not record the CUDA device name. The collector now
records device name, capacity, compute capability, CUDA runtime, and Triton
version. Because the runtime-stride fix changes the compiled kernel, rerun this
short benchmark once after the fix; preserve this artifact as the before-fix
diagnostic.

## Matched 100-request confirmation

All six inputs contain the same 100 seed-42 random QuALITY-dev trace positions,
dataset checksum
`99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef`,
model revision, Torch version, result schema, and segmented-reference checksum.
The analyzer verified contiguous indexes, trace alignment, and zero retained
article KV in the segmented control.

| Path | Article-token hit | Mean TTFT | p50 | p95 | Cache-only speedup | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| Segmented, no document cache | 0.00% | 1.911 s | 2.093 s | 2.644 s | -- | -- |
| Tensor FP16 | 20.31% | 1.612 s | 2.016 s | 2.737 s | 1.186x | [1.076, 1.326] |
| Arena FP16, 64-token pages | 20.31% | 1.701 s | 2.122 s | 2.821 s | 1.124x | [1.021, 1.258] |
| Arena FP16, 256-token pages | 18.19% | 1.665 s | 2.066 s | 2.732 s | 1.148x | [1.049, 1.278] |
| CPU INT8, PyTorch restore | 32.70% | 1.346 s | 1.637 s | 2.586 s | 1.420x | [1.248, 1.652] |
| CPU INT8, Triton restore, diagnostic | 32.70% | 1.453 s | 1.832 s | 2.728 s | 1.315x | [1.161, 1.517] |

The five non-Triton rows are accepted confirmation evidence. Tensor FP16 is
5.54% faster than arena-64 and 3.27% faster than arena-256. That is an expected
and useful negative result: this arena provides deterministic allocation and
lifetime safety, but Transformers still reconstructs contiguous legacy caches
instead of attending directly over page tables.

### Arena accounting

| Metric | Arena 64 | Arena 256 |
|---|---:|---:|
| Reserved slab bytes | 3.999 GiB | 3.999 GiB |
| Peak live allocated bytes | 3.996 GiB | 3.999 GiB |
| Peak stranded tail bytes | 23.54 MiB | 114.35 MiB |
| Arena metadata peak | 37.80 KiB | 10.07 KiB |
| Peak live documents | 26 | 26 |
| Allocations / releases | 81 / 56 | 83 / 58 |
| Stale-handle rejections | 0 | 0 |
| Mean store time per request | 103.06 ms | 26.78 ms |
| Mean restore time per request | 5.50 ms | 1.44 ms |

Smaller pages preserve the tensor backend's hit rate and reduce tail waste, but
require about four times as many page operations. Larger pages reduce store and
restore overhead by about 3.85x and 3.81x respectively, while their extra tail
waste evicts enough useful tokens to lower the hit rate by 2.12 percentage
points.

### Correctness and the Triton diagnostic

Both FP16 arena variants and tensor FP16 agree with the segmented reference on
all 100 labels with zero label-score delta. Both INT8 paths have 99% label
agreement, 50% accuracy versus the reference's 51%, and the same one changed
answer at request index 95 (`20064_CU1CDFL8_6`, article `20064`, reference/gold
`C`, INT8 `A`). PyTorch and Triton INT8 outputs agree exactly with each other
for every A/B/C/D score, showing that the issue below is scheduling/JIT cost,
not a kernel correctness failure.

The initial Triton kernel declared token-dependent `HEAD_STRIDE` as
`tl.constexpr`. The 31 cache hits therefore include a 2.048-second first JIT
compile and roughly 54--86 ms of dequantization on most first-seen article
lengths. Hits whose article length was already compiled take about 1.7--2.7 ms.
Across hits, mean Triton dequantization is 113.23 ms versus 8.02 ms for PyTorch;
this makes the current Triton TTFT row diagnostic and not reportable as the
backend's steady-state performance.

The corrected kernel makes `head_stride` a runtime argument and marks it and
`element_count` as `do_not_specialize`. A one-time model-geometry JIT warm-up
now runs before request timing and is reported separately as
`offline_restore_warmup_s` and its per-request amortization. The corrected
Triton row must preserve all PyTorch-INT8 scores, contain no per-length compile
spikes, and report the warm-up cost outside TTFT before it replaces the
diagnostic row.

## CUDA experiment sequence

After pulling the runtime-stride fix, run the complete suite. The CUDA test now
checks two token lengths against PyTorch, exercising the dynamic stride:

```bash
python -m unittest discover -s tests -v
```

Archive the old diagnostic artifacts before reusing the canonical Triton output
name. The other five confirmation paths must not be rerun:

```bash
mkdir -p results/arena_triton/diagnostic_length_specialized
cp results/arena_triton/confirmation/dev_confirmation_document_triton_random_int8_4gib* \
  results/arena_triton/diagnostic_length_specialized/
```

Rerun the short microbenchmark so it measures the runtime-stride kernel and
writes the expanded hardware manifest:

```bash
python experiments/run_quality.py benchmark-restore \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype float16 \
  --tokens 512 2048 8192 \
  --backends pytorch triton \
  --warmup 2 --repeats 10 --seed 42 \
  --output results/arena_triton/restore_microbenchmark_runtime_stride.csv
```

Then rerun only the corrected Triton confirmation in a fresh process. It reuses
the already validated segmented reference and writes the filename expected by
the analysis suite:

```bash
python experiments/run_quality.py run \
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

Finally validate all six aligned paths and generate the curated tables and
figures:

```bash
python experiments/run_quality.py analyze-inference \
  results/arena_triton/confirmation \
  --suite-config configs/arena_triton_analysis.json \
  --output-dir results/arena_triton/analysis \
  --bootstrap-samples 20000 --seed 42
```

Accept the replacement only if `restore_backends_used` is `["triton"]`, all
100 PyTorch/Triton label scores agree, `reference_label_mismatches` remains one,
and first-seen article lengths no longer show compile-sized dequantization
spikes. Report `offline_restore_warmup_s` separately from TTFT.

## Required interpretation

- Compare tensor FP16 and arena FP16 at identical storage, trace, budget, and
  reference. Report reserved bytes and stranded bytes alongside TTFT.
- Compare PyTorch and Triton INT8 on identical restored requests. Report both
  the microbenchmark and end-to-end TTFT because kernel speedup can be hidden by
  prefill and suffix scoring.
- Require FP16 arena labels to agree with the segmented reference. For INT8,
  report label mismatches, maximum score delta, and accuracy delta rather than
  claiming numerical equivalence.
- Do not merge these rows with schema-v2 archives. They use result schema v3
  and new backend provenance fields.
