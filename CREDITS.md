# Source credits

This branch integrates work from the following projects and contributors. The integration does not claim authorship of their implementations. Original copyright notices and licenses remain intact.

| Work | Source and contributors | Branch / revision used |
| --- | --- | --- |
| Core runtime, GLM5-Next / GLM-5.3-Flash support, MTP, expert cache, and backend improvements | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and its contributors | `master` at `0a1ef309cdbb9f2024f58a38fb4c42352c5db569` |
| RDNA2/V620 kernels, topology policies, and downstream speculative execution work | Edwin Brown and contributors to [edwinbrowwn/llama.cpp-rdna2](https://github.com/edwinbrowwn/llama.cpp-rdna2) | `master` at [`e194cbddce45178467cef4afab0e3400ea357406`](https://github.com/edwinbrowwn/llama.cpp-rdna2/commit/e194cbddce45178467cef4afab0e3400ea357406) |
| Superseded DeepSeek-V4.1 model and conversion support | Victor Cruz (`vcruz305`), `midagedev`, and contributors to [llama.cpp PR 28696](https://github.com/ggml-org/llama.cpp/pull/28696) | [vcruz305/llama.cpp, `feat/deepseek-v41-convert`](https://github.com/vcruz305/llama.cpp/tree/feat/deepseek-v41-convert), snapshot `2076412598e99127ea2554467b2b7530856895d3` |
| Superseded DeepSeek-V4.1 sparse-attention adaptations | `Skelectric` and contributors to [ik_llama.cpp PR 2512](https://github.com/ikawrakow/ik_llama.cpp/pull/2512), built on I. Kawrakow's ik_llama.cpp | [Skelectric/ik_llama.cpp, `ik/deepseek41-switch-b`](https://github.com/Skelectric/ik_llama.cpp/tree/ik/deepseek41-switch-b), snapshot `52b210feb5886a511fb4e868762b97617b4975f5` |
| Multi-GPU MoE prefill distribution and expert-copy work | Paul D. Martin, Ph.D., [llm-performance-engineering-notebook](https://github.com/pauldmartinphd/llm-performance-engineering-notebook) | [`main`, `patches/prefill`](https://github.com/pauldmartinphd/llm-performance-engineering-notebook/tree/main/patches/prefill); adapted as described in [the port notes](docs/moe-prefill-offload.md) |

The historical DeepSeek and sparse-attention sources were adapted, not merged wholesale. They have now been replaced by Daniel Han / Unsloth PR 30817; see [the current integration](docs/deepseek41.md). Branch names identify the source work; pinned revisions identify the imported snapshots even if those branches subsequently change.

The RDNA2 changes were consolidated before rebasing. Its complete original contributor and merge history is available through the linked source revision; this integration's consolidated commit is not a replacement for those credits. The source snapshot includes merged work from branches such as `feat/unified-rdna2-rdna3`, `feat/auto-mtp-stacked-verification`, `feat/qwen38-vocab-sharded-output`, `fix/sidecar-runtime-safety`, `fix/sidecar-structural-match`, and `fix/dflash-cached-turns-snapshot`.

GLM5-Next MTP and expert caching are upstream implementations, not newly authored features of this fork. Relevant upstream changes include [GLM MTP PR 29928](https://github.com/ggml-org/llama.cpp/pull/29928), [expert cache PR 29887](https://github.com/ggml-org/llama.cpp/pull/29887), [multi-GPU cache PR 30112](https://github.com/ggml-org/llama.cpp/pull/30112), and [SLRU cache PR 30637](https://github.com/ggml-org/llama.cpp/pull/30637).

Model quantization repositories are distinct from code sources. The tested or inspected GGUFs came from vcruz305 (DeepSeek target), JigSawPT (DSpark), and Unsloth (GLM target); their artifacts are not included in this repository. See the integration guide for validation limits.
