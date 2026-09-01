"""Synthetic model-geometry benchmark for CPU-INT8 KV restoration."""

from __future__ import annotations

import statistics
import time

from ..reporting import write_csv, write_manifest
from ..reporting.metrics import percentile
from ..schema import RESTORE_BENCHMARK_SCHEMA_VERSION, RESULT_SCHEMA_VERSION
from .restore import resolve_int8_restore_backend, restore_blocks_profiled
from .tensors import KVBlock, QuantizedTensor


def run_restore_benchmark(args) -> int:
    """Run the benchmark without loading model weights.

    ``args`` is the parsed CLI namespace. Keeping command execution here makes
    the CLI an orchestration boundary while this module owns benchmark setup,
    measurement, correctness checks, and artifact metadata.
    """
    import torch
    from transformers import AutoConfig

    if not torch.cuda.is_available():
        raise RuntimeError("benchmark-restore requires an allocated CUDA GPU")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    config = AutoConfig.from_pretrained(args.model, revision=args.model_revision)
    layers = int(config.num_hidden_layers)
    kv_heads = int(
        getattr(config, "num_key_value_heads", config.num_attention_heads)
    )
    head_dim = int(
        getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
    )
    resolved_backends = {
        backend: resolve_int8_restore_backend(backend, device)
        for backend in args.backends
    }
    model_revision = str(
        getattr(config, "_commit_hash", None)
        or args.model_revision
        or args.model
    )

    torch.manual_seed(args.seed)
    rows = []
    for token_count in args.tokens:
        block = _make_quantized_block(
            torch,
            layers=layers,
            kv_heads=kv_heads,
            head_dim=head_dim,
            token_count=token_count,
        )
        reference = restore_blocks_profiled(
            [block], dtype=dtype, device=device, int8_backend="pytorch"
        ).cache
        for requested_backend in args.backends:
            row = _benchmark_backend(
                torch,
                block=block,
                reference=reference,
                requested_backend=requested_backend,
                device=device,
                dtype=dtype,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            row.update(
                {
                    "benchmark_schema_version": RESTORE_BENCHMARK_SCHEMA_VERSION,
                    "result_schema_version": RESULT_SCHEMA_VERSION,
                    "model": args.model,
                    "model_revision": model_revision,
                    "device": args.device,
                    "dtype": args.dtype,
                    "seed": args.seed,
                    "tokens": token_count,
                    "layers": layers,
                    "kv_heads": kv_heads,
                    "head_dim": head_dim,
                    "backend_requested": requested_backend,
                    "backend_resolved": resolved_backends[requested_backend],
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                }
            )
            rows.append(row)
        del reference, block
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, rows)
    write_manifest(
        args.output,
        {
            "run_type": "int8-restore-microbenchmark",
            "benchmark_schema_version": RESTORE_BENCHMARK_SCHEMA_VERSION,
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "model": args.model,
            "model_revision": model_revision,
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
            "tokens": args.tokens,
            "backends_requested": args.backends,
            "backends_resolved": resolved_backends,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "synthetic_inputs": True,
        },
    )
    print(f"wrote {len(rows)} restore benchmark rows to {args.output}")
    return 0


def _make_quantized_block(
    torch,
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    token_count: int,
) -> KVBlock:
    shape = (1, kv_heads, token_count, head_dim)
    quantized_layers = []
    for _ in range(layers):
        key_values = torch.randint(-127, 128, shape, dtype=torch.int8)
        value_values = torch.randint(-127, 128, shape, dtype=torch.int8)
        key_scales = torch.rand(kv_heads, dtype=torch.float32).clamp_min_(1e-5)
        value_scales = torch.rand(kv_heads, dtype=torch.float32).clamp_min_(1e-5)
        quantized_layers.append(
            (
                QuantizedTensor(key_values, key_scales),
                QuantizedTensor(value_values, value_scales),
            )
        )
    return KVBlock(tuple(quantized_layers), token_count)


def _benchmark_backend(
    torch,
    *,
    block: KVBlock,
    reference,
    requested_backend: str,
    device,
    dtype,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | str]:
    for _ in range(warmup):
        warmup_result = restore_blocks_profiled(
            [block],
            dtype=dtype,
            device=device,
            int8_backend=requested_backend,
        )
        del warmup_result

    elapsed_values = []
    load_values = []
    transfer_values = []
    dequant_values = []
    last = None
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        started = time.perf_counter()
        result = restore_blocks_profiled(
            [block],
            dtype=dtype,
            device=device,
            int8_backend=requested_backend,
        )
        elapsed_values.append(time.perf_counter() - started)
        load_values.append(result.load_s)
        transfer_values.append(result.transfer_s)
        dequant_values.append(result.dequant_s)
        if last is not None:
            del last
        last = result

    max_error = max(
        float((actual - expected).abs().max().item())
        for actual_layer, expected_layer in zip(last.cache, reference)
        for actual, expected in zip(actual_layer, expected_layer)
    )
    restored_bytes = sum(
        tensor.numel() * tensor.element_size()
        for layer in last.cache
        for tensor in layer
    )
    mean_restore = statistics.fmean(elapsed_values)
    row = {
        "stored_int8_bytes": block.stored_bytes,
        "restored_bytes": restored_bytes,
        "restore_mean_s": mean_restore,
        "restore_p50_s": percentile(elapsed_values, 0.50),
        "restore_p95_s": percentile(elapsed_values, 0.95),
        "transfer_mean_s": statistics.fmean(transfer_values),
        "dequant_mean_s": statistics.fmean(dequant_values),
        "load_mean_s": statistics.fmean(load_values),
        "stored_gib_per_s": block.stored_bytes / max(mean_restore, 1e-12) / 2**30,
        "max_abs_error_vs_pytorch": max_error,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    del last, result
    return row
