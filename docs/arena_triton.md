# Document KV arena and Triton restore

This phase evaluates two optional physical backends without replacing the
frozen Qwen2.5-1.5B or Qwen2.5-0.5B baseline claims. The final aligned
100-request confirmation is complete. A first Triton trace exposed
token-length-dependent JIT specialization; the corrected kernel makes the
stride runtime-valued and prewarms one model-geometry kernel outside request
timing. The replacement passes score parity, latency-spike, allocator, trace,
and provenance checks. Curated tables and figures are frozen in
[`generated/arena_triton`](generated/arena_triton/README.md).

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

The final corrected synthetic-geometry benchmark uses Qwen2.5-1.5B geometry:
28 layers, two KV heads, head dimension 128, FP16 output, seed 42, two warm-ups,
and ten measured repetitions. Both requested backends resolved to themselves.
The run records a Tesla T4, Torch `2.11.0+cu128`, CUDA 12.8, Triton 3.6.0,
model revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, result schema
`quality-kv-v3`, and Git revision
`710037420045ebc08a4ccd9d28dc1e8ae8b36420`.

| Tokens | PyTorch restore | Triton restore | Restore speedup | PyTorch dequant | Triton dequant | Dequant speedup | Transfer share of Triton restore |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 6.120 ms | 5.381 ms | 1.137x | 2.260 ms | 1.496 ms | 1.510x | 70.49% |
| 2,048 | 13.595 ms | 11.655 ms | 1.167x | 3.162 ms | 1.564 ms | 2.022x | 85.74% |
| 8,192 | 43.387 ms | 34.915 ms | 1.243x | 11.281 ms | 2.190 ms | 5.150x | 93.37% |

Stored-byte throughput rises from 1.117 to 1.270 GiB/s at 512 tokens, from
2.011 to 2.346 GiB/s at 2,048, and from 2.521 to 3.133 GiB/s at 8,192.
`max_abs_error_vs_pytorch` is exactly zero in all six rows. An independent
second corrected repetition gives restore-speedup ranges of 1.137--1.138x,
1.167--1.186x, and 1.243--1.267x at the three token lengths, also with exact
output parity.

The canonical CSV SHA-256 is
`f021911cdb38a838839da5649ec060f7665b745b1f054c02e0441039c732f474`;
the independent repetition is
`6ba5330da954a3d2ba179743d0539390dff82463bbdd7a03af9ba6c3e288a0ac`.
Both CSVs and their complete manifests are preserved in
[`generated/arena_triton`](generated/arena_triton/README.md).

This supports a deliberately narrow kernel claim. Triton reduces fused
dequantization/cast time and improves restore throughput with exact synthetic
parity. It does not remove host-to-device movement: transfer accounts for
93.37% of Triton restore at 8,192 tokens. Pinned asynchronous transfer and
overlap are plausible future work, but require stream-aware lifetime tests and
a separate experiment.

## Matched 100-request confirmation

All six final inputs contain the same 100 seed-42 random QuALITY-dev trace
positions, dataset checksum
`99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef`,
model revision, Torch version, result schema, code revision, and segmented-
reference checksum. The analyzer verified contiguous indexes, trace alignment,
and zero retained article KV in the segmented control.

| Path | Article-token hit | Mean TTFT | p50 | p95 | Cache-only speedup | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|
| Segmented, no document cache | 0.00% | 1.941 s | 2.095 s | 2.605 s | -- | -- |
| Tensor FP16 | 20.31% | 1.544 s | 1.944 s | 2.532 s | 1.257x | [1.141, 1.406] |
| Arena FP16, 64-token pages | 20.31% | 1.723 s | 2.164 s | 2.936 s | 1.126x | [1.024, 1.260] |
| Arena FP16, 256-token pages | 18.19% | 1.687 s | 2.097 s | 2.759 s | 1.150x | [1.051, 1.280] |
| CPU INT8, PyTorch restore | 32.70% | 1.438 s | 1.708 s | 2.848 s | 1.350x | [1.186, 1.573] |
| CPU INT8, Triton restore | 32.70% | 1.418 s | 1.671 s | 2.787 s | 1.369x | [1.202, 1.589] |

Tensor FP16 is 11.62% faster than arena-64 and 9.30% faster than arena-256.
That is a useful negative result: the arena provides deterministic allocation
and lifetime safety, but Transformers still reconstructs contiguous legacy
caches instead of attending directly over page tables. The Triton row is valid
online evidence after the runtime-stride correction, subject to the explicit
startup qualification below.

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
| Mean store time per request | 105.03 ms | 28.40 ms |
| Mean restore time per request | 5.58 ms | 1.43 ms |

Smaller pages preserve the tensor backend's hit rate and reduce tail waste, but
require about four times as many page operations. Larger pages reduce store and
restore overhead by about 3.70x and 3.90x respectively, while their extra tail
waste evicts enough useful tokens to lower the hit rate by 2.12 percentage
points. Arena-256 is 2.08% faster than arena-64 but still slower than tensor
storage.

### Correctness and corrected Triton comparison

Both FP16 arena variants and tensor FP16 agree with the segmented reference on
all 100 labels with zero label-score delta. Both INT8 paths have 99% label
agreement, 50% accuracy versus the reference's 51%, and the same one changed
answer at request index 95 (`20064_CU1CDFL8_6`, article `20064`, reference/gold
`C`, INT8 `A`). PyTorch and Triton INT8 outputs agree exactly with each other
for all 100 labels and all A/B/C/D scores. Against FP16, their maximum label-
score delta is 0.828125, their mean per-request maximum delta is 0.085625, and
30 restored requests exceed the strict 0.0625 tolerance. These are numerical
differences, not 30 answer errors.

The matched comparison contains 31 cache hits:

| Hit-path component | PyTorch INT8 | Triton INT8 | PyTorch / Triton | Reduction |
|---|---:|---:|---:|---:|
| TTFT | 94.17 ms | 86.73 ms | 1.086x | 7.90% |
| Restore | 37.45 ms | 30.98 ms | 1.209x | 17.27% |
| Transfer | 29.13 ms | 28.62 ms | 1.018x | 1.74% |
| Dequantization | 8.06 ms | 2.15 ms | 3.754x | 73.36% |

The request-level paired bootstrap 95% interval for hit-only restore speedup is
[1.174, 1.243]. This is the defensible end-to-end kernel-path result: transfer
remains dominant, but fused dequantization lowers restore and hit TTFT.

The corrected kernel makes `head_stride` runtime-valued and prevents token-
length specialization. The earlier diagnostic trace had 113.23 ms mean hit
dequantization and a 2.048-second spike; the corrected mean is 2.15 ms, 52.8x
lower, with no per-length compile spike. That diagnostic remains useful as the
bug-finding history but is excluded from final timing tables.

### Startup amortization

Triton's one-time model-geometry warm-up is 2.671 seconds and is reported as
`offline_restore_warmup_s`, outside request TTFT. Online mean TTFT is 1.418
seconds for Triton versus 1.438 seconds for PyTorch, a small 1.4% difference.
That all-request difference is secondary: miss/store timing varied between the
two independent runs, so it cannot be attributed entirely to the restore
kernel. Amortizing warm-up over only 100 requests raises Triton to 1.445
seconds, about 0.49% slower than PyTorch. Using only the observed hit-restore
savings, the warm-up breaks even after roughly 1,332 total requests at this 31%
hit rate. Triton is therefore a long-lived-server optimization, not a win for a
short one-shot job.

## CUDA experiment sequence

The final evidence is frozen, so these commands are for reproduction on a
fresh CUDA runtime rather than required follow-up work. First validate the
implementation; the CUDA test checks two token lengths against PyTorch and the
CPU suite exercises allocator lifetime and fallback behavior:

```bash
python -m unittest discover -s tests -v
```

Run the short corrected microbenchmark with complete hardware provenance:

```bash
python experiments/run_quality.py benchmark-restore \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --device cuda --dtype float16 \
  --tokens 512 2048 8192 \
  --backends pytorch triton \
  --warmup 2 --repeats 10 --seed 42 \
  --output results/arena_triton/restore_microbenchmark_runtime_stride.csv
```

Run the complete six-path confirmation. `--resume` skips completed JSONLs and
each path gets a neighboring manifest:

```bash
python experiments/run_quality.py matrix configs/arena_triton_confirmation.json \
  --profile confirmation --execute --resume
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

The reproduced suite should contain six aligned 100-request rows, exact FP16
agreement, exact PyTorch/Triton INT8 agreement, one shared INT8 label mismatch
against FP16, no stale-handle rejection, and no token-length compile spikes.
Always report `offline_restore_warmup_s` separately from online TTFT. The exact
final CSVs, figures, hashes, and machine provenance are in
[`generated/arena_triton`](generated/arena_triton/README.md).

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
