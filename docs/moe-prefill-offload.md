# Multi-GPU MoE prefill offload

Adapted from Paul Martin's [prefill notebook patch](https://github.com/pauldmartinphd/llm-performance-engineering-notebook/blob/main/patches/prefill/README.md). His July GLM-5.2 experiment measured 13.7% better pp8192 throughput from distribution plus routing-readback bypass; distribution alone did not improve throughput. Those results do not establish performance for this port or other models. Codex adapted and locally tested this port.

The scheduler distributes host-weight MUL_MAT_ID operations across eligible backends by layer number, keeping gate/up/down on the same backend when their eligible sets match. Unnamed tensors use a round-robin fallback reset for each graph split pass. Only batches of at least 32 tokens use this policy; backend support and offload eligibility still apply. Non-MoE operations keep their original priority order.

The expert-copy callback now lives in llama-context.cpp. At 32 or more tokens and at least eight routing selections per model expert, it copies all experts without synchronously reading routing IDs. This is a transfer heuristic: skewed routing can copy unused experts. Smaller batches retain selective copying. Existing cache hits are usable only on the cache's owning device; other destinations upload from RAM as before.

Both changes are enabled by default. Set `GGML_MOE_PREFILL=0` before starting the process to disable both for an A/B comparison. This is separate from `--moe-cache-mib`. Start with that cache disabled. These changes distribute streamed work; they do not change persistent model-weight placement or enable tensor parallelism.

Local validation: all configured CPU build targets pass; an extracted-source harness checks four-backend selection, eligible subsets, gate/up/down affinity, fallback, non-MoE and decode placement, routing threshold boundaries, no ID reads on the full-copy path, ID reuse on the selective path, and numerical equivalence with deliberately skewed routing using the real CPU backend. Synthetic GLM5-Next and DeepSeek-V4.1 tests and five split/speculative regressions pass. HIP compilation and performance of this port remain untested locally.
