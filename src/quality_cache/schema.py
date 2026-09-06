"""Version identifiers for result and calibration artifacts."""

# v4 includes nested, identity-deduplicated arena allocator metadata. Legacy
# v3 arena metadata and metadata-inclusive footprints are shallow estimates.
RESULT_SCHEMA_VERSION = "quality-kv-v4"
PREFILL_CALIBRATION_SCHEMA_VERSION = "quality-prefill-cost-v1"
RESTORE_BENCHMARK_SCHEMA_VERSION = "quality-int8-restore-benchmark-v1"
INFERENCE_TIMING_SCOPE = "model-forward-excludes-tokenization-v1"
NO_INFERENCE_TIMING_SCOPE = "simulated-no-inference-v1"
