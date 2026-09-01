# Code-quality and architecture review

This review was performed after the frozen Qwen2.5-1.5B baseline and the
matched-working-set Qwen2.5-0.5B confirmation. The result is a bounded
refactor: experiment semantics and public flags are unchanged, while physical
allocation, tensor representation, restore strategy, and command execution now
have separate owners.

## Current assessment

The repository is appropriate for a research course project. The core cache
implementations already have several properties that are often missing from
prototype code:

- one structural cache protocol and a registry-based factory;
- hard tensor-byte bounds and explicit metadata accounting;
- bounded lazy eviction heaps instead of full scans on every eviction;
- deterministic workload construction and reference alignment by trace
  position;
- strategy-specific modules rather than one cache implementation with many
  branches; and
- tests for cache identity, reachability, stale handles, numerical agreement,
  and a 20,000-request inference-free trace.

The main maintainability issue was responsibility mixing, not a flawed cache
algorithm. Tensor storage also performed restoration, the arena combined page
lifetime rules with slab access, and the unified CLI contained the complete
CUDA benchmark. Those boundaries are now explicit.

## Refactor applied

| Responsibility | Module | Pattern or boundary |
|---|---|---|
| Common KV shape and stored-block contract | `inference/kv_types.py` | Structural protocol; caches depend on capabilities rather than `KVBlock` |
| Object tensor storage, INT8 representation, and slicing | `inference/tensors.py` | Data representation |
| Restore timing and backend selection | `inference/restore.py` | Strategy registry with PyTorch and Triton callables |
| Page ownership and generation validation | `inference/page_allocator.py` | Resource allocator independent of PyTorch |
| Physical FP16 slab reads and writes | `inference/arena.py` | Storage owner composed with `PageAllocator` |
| Synthetic restore experiment | `inference/restore_benchmark.py` | Command handler outside the parser/orchestrator |

The allocator keeps deterministic lowest-page allocation, but now validates
that free and allocated page sets are unique, disjoint, complete, and agree
with live byte counters. Malformed, foreign, released, and reused handles are
rejected before any slab access. These invariants can be tested without
allocating a tensor. Normal allocation performs constant-time counter checks;
the exhaustive O(total-pages) ownership audit is explicit test/debug work, not
part of every serving operation.

The restore module uses a small function registry instead of a class hierarchy.
That is the Strategy pattern at the scale needed here: adding a backend means
implementing one pair-dequantization callable and registering it. Introducing
dependency-injection containers or a policy class per LRU/LFU/GDSF formula
would add ceremony without providing a second implementation boundary.

The first end-to-end Triton trace also caught a systems-specific scalability
bug that an isolated fixed-shape benchmark could not reveal: token-dependent
`HEAD_STRIDE` was a compile-time specialization, causing a new JIT kernel for
nearly every article length. It is now runtime-valued and explicitly excluded
from specialization. One model-geometry compile occurs before the request loop,
and its cost is reported separately rather than hidden in TTFT. This is why the
project retains both microbenchmarks and variable-length end-to-end traces.

The cleanup also removes the unused standalone `dequantize` function and
unused concrete-block imports. Existing factories and bounded heaps were kept;
rewriting working structures merely to name another design pattern would make
the empirical code harder to audit.

## Relationship to inference engines

The design borrows boundaries, not serving-stack code:

| Engine | Relevant production design | This repository |
|---|---|---|
| SGLang | Separates request/token mapping, page allocation, physical KV pools, and radix ownership in [`memory_pool.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py), [`paged.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/allocator/paged.py), and [`radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) | Separates document policy, `PageAllocator`, and `KVArena`; document eviction remains atomic |
| vLLM | Preallocates fixed blocks and tracks free blocks, reference counts, and prefix hashes in [`block_pool.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py) | Uses fixed blocks as a generic baseline, but keeps the proposed cache unit aligned with known article boundaries |
| TensorRT-LLM | Manages a pool of fixed-token blocks with reuse, offload, and prioritized eviction in its [KV-cache documentation](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/features/kvcache.md) | Measures residency and CPU INT8 restoration explicitly, without claiming a scheduler or page-aware attention kernel |

The important limitation remains: Transformers attention consumes a contiguous
legacy cache. The arena must assemble pages into a transient contiguous cache
on a hit. SGLang and other serving engines can make attention consume indexed
locations directly. Reaching that level would require an attention-runtime and
scheduler project, not another allocator refactor.

## Deliberate non-changes

- LRU, LFU, FIFO, and GDSF remain compact policy functions. Their shared state
  is small, and the existing deterministic tests make the formulas auditable.
- Fixed-block and radix keep their tuned bounded heaps. A shared generic tree
  framework would hide strategy-specific lifetime rules.
- `ModelRunner` was not broadly split because its timing boundaries are part
  of the experiment schema. A cosmetic rewrite would risk invalidating
  comparison semantics.
- Historical configs, generated evidence, and raw ignored results were not
  deleted. They are provenance, not dead source code.

## Remaining engineering debt

The next useful cleanup is command-level extraction of calibration, simulation,
and inference handlers if `cli.py` grows again. It is not required for the
current matrix. For a production server, the larger missing pieces are
concurrent request scheduling, asynchronous/pinned host transfer, stream-aware
lifetime management, and page-table-aware attention. They should be described
as out of scope rather than simulated with extra abstraction.

Repository handoff should continue to require the complete CPU suite,
`git diff --check`, matrix-expansion tests, and a CUDA smoke run for changes to
arena or Triton code.
