# Frozen arena and Triton evidence

This directory contains the curated analysis generated from the final aligned
Qwen2.5-1.5B arena/Triton confirmation. Raw JSONLs remain outside Git; their
SHA-256 hashes and Git revisions are recorded in [`analysis.json`](analysis.json).

## Provenance

| Field | Value |
|---|---|
| Date checked | 1 September 2026 |
| Dataset | QuALITY v1.0.1 HTML-stripped dev |
| Dataset SHA-256 | `99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef` |
| Trace | 100 random requests, seed 42 |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Model revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Code revision | `710037420045ebc08a4ccd9d28dc1e8ae8b36420` |
| Result schema | `quality-kv-v3` |
| Analysis schema | `quality-fair-inference-v2` |
| Hardware | Tesla T4, 15,637,086,208 bytes, compute capability 7.5 |
| Software | Torch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0 |

The analyzer verified six runs, contiguous indexes, identical request traces,
the segmented-control invariants, and a common dataset checksum. The five
cached rows use the segmented JSONL with SHA-256
`3fe1c169200cc85d0a9a57ecc334d50e7c9c7d68b7a39ef5f339d43dbb1f7f64`
as their offline reference.

## Contents

- [`results.md`](results.md) is the generated fair-comparison summary.
- [`run_summaries.csv`](run_summaries.csv) contains aggregate latency, cache,
  allocator, and correctness measurements.
- [`fair_speedups.csv`](fair_speedups.csv) contains cache-only speedups and
  20,000-sample paired bootstrap intervals.
- [`correctness.csv`](correctness.csv) and
  [`mismatch_details.json`](mismatch_details.json) contain label/score checks.
- [`triton_comparison.csv`](triton_comparison.csv) isolates the matched
  PyTorch-versus-Triton hit path and startup amortization.
- The PNG/PDF files are the generated report figures.
- The two restore CSVs are independent repetitions on the corrected runtime-
  stride kernel. Both manifests record the complete GPU/software provenance.

The speedup figure was regenerated from the byte-identical archived analysis
tables after a presentation-only fix made backend labels explicit. No measured
value, confidence interval, input hash, or generated results table changed.

## Interpretation guardrail

The online Triton row excludes a separately reported 2.671-second one-time JIT
warm-up. Amortizing it over only 100 requests makes Triton slightly slower than
PyTorch INT8. The kernel claim should therefore use hit-only restore and the
restore microbenchmark; the 1.4% online mean-TTFT difference is secondary and
must not be attributed entirely to the kernel because miss/store timing also
varied between runs.
