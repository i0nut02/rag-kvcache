# Document KV arena and Triton restore

This phase adds two optional physical backends without changing the frozen
Qwen2.5-1.5B or Qwen2.5-0.5B claims. The restore-only CUDA microbenchmark is
complete and shows a Triton benefit. The matched end-to-end arena/Triton matrix
is still required before making a TTFT claim for either backend.

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

The first Qwen2.5-1.5B synthetic-geometry run used model revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, 28 layers, two KV heads,
head dimension 128, FP16 output, seed 42, two warm-ups, and ten measured
repetitions. Both requested backends resolved to themselves. The benchmark
uses schema `quality-int8-restore-benchmark-v1` and result schema
`quality-kv-v3`.

| Tokens | PyTorch restore | Triton restore | Restore speedup | PyTorch throughput | Triton throughput | Triton p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 6.544 ms | 5.696 ms | 1.149x | 1.045 GiB/s | 1.200 GiB/s | 5.917 ms |
| 2,048 | 14.842 ms | 11.910 ms | 1.246x | 1.842 GiB/s | 2.296 GiB/s | 12.212 ms |
| 8,192 | 44.102 ms | 34.572 ms | 1.276x | 2.480 GiB/s | 3.164 GiB/s | 36.935 ms |

The mean restore reductions are 12.96%, 19.76%, and 21.61% as the restored
prefix grows. The kernel-only dequantization result explains that scaling:

| Tokens | PyTorch dequant | Triton dequant | Dequant speedup | Triton transfer | Transfer share of Triton restore |
|---:|---:|---:|---:|---:|---:|
| 512 | 2.353 ms | 1.568 ms | 1.500x | 4.029 ms | 70.73% |
| 2,048 | 3.403 ms | 1.500 ms | 2.268x | 10.299 ms | 86.48% |
| 8,192 | 11.134 ms | 1.723 ms | 6.461x | 32.727 ms | 94.66% |

`max_abs_error_vs_pytorch` is exactly zero in all six rows. Peak CUDA
allocation is also slightly lower for Triton: 49.03 versus 49.78 MiB at 512
tokens, 196.03 versus 199.03 MiB at 2,048, and 784.03 versus 796.03 MiB at
8,192.

This supports a narrow claim: the fused Triton dequantization/cast is correct
for these inputs and improves isolated restore throughput. It does not yet
show an end-to-end TTFT improvement. At 8,192 tokens the measured transfer is
94.66% of Triton restore time, so host-to-device movement is now the dominant
bottleneck. Pinned/asynchronous transfer and overlap are possible future work,
but require stream-aware lifetime tests and a separate experiment.

These values were transcribed from the supplied benchmark table. The original
CSV and neighboring manifest, including the CUDA device name and code revision,
must be retained with the final artifact archive; the hardware fields were not
present in the table pasted into this repository review. Because the subsequent
code-quality refactor moved restore orchestration into its own module, run this
short benchmark once more from the final submitted commit before treating the
numbers as final report evidence. The existing run remains useful validation of
the kernel and expected trend.

## CUDA experiment sequence

After pulling the implementation, run the complete CPU suite and validate the
six-run matrix without loading weights:

```bash
python -m unittest discover -s tests -v
python experiments/run_quality.py matrix \
  configs/arena_triton_confirmation.json \
  --profile smoke --show-commands
```

To reproduce the synthetic model-geometry microbenchmark on the final commit,
use the following command. It excludes model weights, warms up Triton
compilation, compares every result with the PyTorch output, and writes a
manifest:

```bash
python experiments/run_quality.py benchmark-restore \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype float16 \
  --tokens 512 2048 8192 \
  --backends pytorch triton \
  --warmup 2 --repeats 10 --seed 42 \
  --output results/arena_triton/restore_microbenchmark_final.csv
```

The next unfinished step is ten requests to validate allocation, kernel
compilation, logits, and GPU memory before the 100-request confirmation:

```bash
python experiments/run_quality.py matrix \
  configs/arena_triton_confirmation.json \
  --profile smoke --execute --resume
```

Inspect all six smoke summaries first. If allocation, memory, and reference
checks pass, run the confirmation:

```bash
python experiments/run_quality.py matrix \
  configs/arena_triton_confirmation.json \
  --profile confirmation --execute --resume
```

The matrix contains one segmented random reference plus tensor FP16, arena
FP16 with 64- and 256-token pages, PyTorch CPU INT8, and Triton CPU INT8. All
five cached runs use the same 100 trace positions and saved reference JSONL.
Smoke rows are functional checks and must not be reported as timing evidence.

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
