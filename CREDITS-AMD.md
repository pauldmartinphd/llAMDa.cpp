# AMD integration sources

This branch builds on ggml-org/llama.cpp and its contributors. Upstream history and licenses remain intact. Imported changes retain the source project's license.

| Source | Revision | Included work |
| --- | --- | --- |
| [Edwin Brown's RDNA2 fork](https://github.com/edwinbrowwn/llama.cpp-rdna2) | `e194cbddce45178467cef4afab0e3400ea357406`, as integrated in this repository's `v620-dev` at `6a0eaddd72b6ecca72f3d0ac004d7e018547a607` | Gfx1030 native wave reductions for tiled flash attention, GDN broadcast loads, exact NVFP4 scale decode, gfx1030/gfx1100 MXFP4 vector scheduling and gfx1100 Q8_0 vector scheduling. |
| [milpster/gfx906-llama-cpp](https://github.com/milpster/gfx906-llama-cpp) | `1373e889a4d5c69f453fe82c96332585f3d2b960` | Vega 20 MMQ table, selected for gfx906 in host and device dispatch. |
| [Strix-Halo/strix-llama.cpp](https://github.com/Strix-Halo/strix-llama.cpp/commit/1674aa4b4b70b1482c21a1b3881c4506191c4f6c) | `1674aa4b4b70b1482c21a1b3881c4506191c4f6c` | Tiled transpose/concat shader and shape dispatch, adapted to upstream's split Vulkan sources and restricted to AMD devices. |

Already present in the upstream base, and deliberately not duplicated:

- [gfx900 MMQ dispatch, PR #24588](https://github.com/ggml-org/llama.cpp/pull/24588), commit `d9df11006f6d3cc34772379bf6897f892874656f`.
- [RDNA3.5 MMQ tile tuning, PR #28195](https://github.com/ggml-org/llama.cpp/pull/28195), commit `4dd6dd4b96f32eb53a91d659bbff8b37bbd282b9`.
- [Vulkan IQ3_S batched matrix-vector tuning, PR #27449](https://github.com/ggml-org/llama.cpp/pull/27449), commit `ba8818cbf3ad2f27f6b50e85b959ada4734f34c3`.

Paul D. Martin maintains this integration. Architecture selection uses the backend's device identity or compiler target. Newer upstream Q2_0 MMQ support is retained on gfx906. Features coupled to custom launchers, provider bundles, activation-cache layouts, or topology experiments have not been copied wholesale.
