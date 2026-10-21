# Multi-GPU MoE prefill offload

Adapted from Paul Martin's [prefill notebook patch](https://github.com/pauldmartinphd/llm-performance-engineering-notebook/blob/main/patches/prefill/README.md). His July GLM-5.2 experiment measured 13.7% better pp8192 throughput from distribution plus routing-readback bypass; distribution alone did not improve throughput. Those results do not establish performance for this port or other models.

The scheduler distributes host-weight MUL_MAT_ID operations across eligible backends by layer number, keeping gate/up/down on the same backend when their eligible sets match. Unnamed tensors use a round-robin fallback reset for each graph split pass. Only batches of at least 32 tokens use this policy; backend support and offload eligibility still apply. Non-MoE operations keep their original priority order.

The expert-copy callback now lives in llama-context.cpp. At 32 or more tokens and at least eight routing selections per model expert, it copies all experts without synchronously reading routing IDs. This is a transfer heuristic: skewed routing can copy unused experts. Smaller batches retain selective copying. Existing cache hits are usable only on the cache's owning device; other destinations upload from RAM as before.

Both changes are enabled by default. Set `GGML_MOE_PREFILL=0` before starting the process to disable both for an A/B comparison. This is separate from `--moe-cache-mib`. Start with that cache disabled. These changes distribute streamed work; they do not change persistent model-weight placement or enable tensor parallelism.

Validation uses `python3 tests/test-moe-prefill-port.py <source-dir> <CPU-build-dir>`. The harness extracts the actual selection and expert-copy code, mocks device eligibility, and checks copied expert results with the real CPU backend. It covers four-device placement, eligible subsets, layer affinity, unnamed-tensor fallback, decode/non-MoE behavior, the bypass threshold, skewed routing and the disable switch. It does not establish GPU transfer overlap or a throughput gain on the current models.
