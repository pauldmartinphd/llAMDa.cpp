# Vega integration

Source: [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) at `c4d12e2c0094c9028318bed5bb1a30bb48f8a35b`.

Common base: `amd-main`, including `fc47f122ea9def218e3ebc6efa6fbc1d456a9771`. The source branch is merged with its history intact so existing authors retain credit and later updates have a traceable merge base.

## Conflict resolutions

The integration retains upstream vectorized normalization and scheduler construction while adding the source input synchronization policy. It preserves upstream model buffer allocation and chunked-read error checks, incorporates PLE prefaulting, and adapts speculative probability bookkeeping to upstream batch ownership. The source hardware tuning targets MI50, MI60 and Radeon VII (gfx906); this is not evidence of equivalent gains on Vega 56/64 or MI25 (gfx900).

The common normalization fix removes a stale launch-wrapper fragment left by an earlier conflict resolution. The source loop specializations and newer upstream vectorized dispatch remain.

## Validation

The full CPU build passed on macOS with Metal disabled. Targeted CPU tests cover chat, sampling, GGUF, quantization, allocation and related changed components. These checks do not compile HIP or Vulkan, validate GPU kernels, or establish inference correctness or performance on AMD hardware. Hardware validation remains required before promoting further changes to `amd-main`.
