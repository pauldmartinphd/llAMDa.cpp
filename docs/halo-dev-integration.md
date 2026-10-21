# Halo integration

Source: [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) at `da731374077f2ad9bdffd4848f782b5694b9bb68`.

Common base: `amd-main`, including `fc47f122ea9def218e3ebc6efa6fbc1d456a9771`. The source branch is merged with its history intact so existing authors retain credit and later updates have a traceable merge base.

## Conflict resolutions

The integration retains upstream device initialization and raw BF16 KV handling alongside the source Vulkan attention paths. It keeps the source complete-block QSA implementation, adapts speculative replay to upstream batch ownership, and preserves upstream audio and TTS handling. Optional PLE disk prefetch is disabled on platforms without POSIX_FADV_WILLNEED; regular reads remain available.

The common normalization fix removes a stale launch-wrapper fragment left by an earlier conflict resolution. The source loop specializations and newer upstream vectorized dispatch remain.

## Validation

The full CPU build passed on macOS with Metal disabled. Targeted CPU tests cover chat, sampling, GGUF, quantization, allocation and related changed components. These checks do not compile HIP or Vulkan, validate GPU kernels, or establish inference correctness or performance on AMD hardware. Hardware validation remains required before promoting further changes to `amd-main`.

## Repository automation

GitHub Actions workflows remain identical to amd-main. Source-fork CI deployment and hardware-runner configuration are not adopted by this integration. The source commits remain in history; their workflow changes are explicitly reverted at the branch tip.

## Unsloth rebase

This branch is rebased onto the shared Unsloth V4.1 integration. The DeepSeek-V4.1 model, DSpark decoder and converter now come from amd-main; previous source-fork V4.1 implementations are superseded. See [the current integration](deepseek41.md) for model compatibility and known issues.
