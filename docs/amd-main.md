# AMD main branch

`amd-main` is based on upstream `0a1ef309cdbb9f2024f58a38fb4c42352c5db569`. It carries a selected set of automatic backend changes. It does not replace `v620-dev`'s full integration: DeepSeek-V4.1/DSpark ports, custom expert-cache and prefill controls, sidecar bundles, and topology experiments remain on that branch. Model support here follows this upstream base.

## Automatic paths

| Hardware | Backend | Changes |
| --- | --- | --- |
| Vega 56/64, MI25 (gfx900) | HIP | Existing upstream gfx900 dispatch. No gfx906 table is applied to gfx900. |
| Radeon VII, MI50/MI60 (gfx906) | HIP | Vega 20 MMQ occupancy/tile table; upstream Q2_0 configuration retained. |
| V620 and other gfx1030 devices | HIP | Native wave reductions in tiled flash attention, broadcast loads during GDN prefill, MXFP4 matrix-vector scheduling, exact NVFP4 scale conversion. |
| gfx1100 | HIP | MXFP4 and Q8_0 matrix-vector scheduling. |
| Strix Halo (gfx1151) | Vulkan / HIP | AMD Vulkan tiled transpose/concat for eligible F32 shapes; upstream IQ3_S batched decoding and HIP RDNA3.5 MMQ tuning retained. |

Dispatch is automatic for the applicable device, operation, shape, and quantization. A model that does not execute these operations will not benefit from those paths. GDN and transpose/concat changes mainly concern hybrid models; they are not general DeepSeek/GLM decode speedups. Native reduction builtins fall back to the existing shuffle implementation when unavailable in the compiler.

No custom build or launch scripts are added. No `GGML_HIP_RDNA2_AUTO`, `SPEC_SIDECAR`, or architecture override is required by these additions. Normal upstream scripts remain in the repository. Backend support for each GPU still depends on the installed compiler/runtime/driver.

## Build

Use a separate build directory for each backend/target. For V620:

```bash
cmake -S . -B build-gfx1030 -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release
cmake --build build-gfx1030 --config Release -j
```

For gfx900, gfx906, gfx1100, or gfx1151 HIP, substitute that target and build-directory name. A ROCm toolchain that can compile and run the requested architecture is required, particularly for older Vega GPUs.

For Strix Halo Vulkan, with Vulkan development packages and a shader compiler installed:

```bash
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j
```

Run the resulting binaries with ordinary llama.cpp options. For example:

```bash
./build-gfx1030/bin/llama-server -m /path/to/model.gguf -ngl 99 -fa on --host 127.0.0.1 --port 8080
```

Model placement, expert offload, and multi-GPU splitting remain explicit runtime choices. This branch does not infer a four-card configuration.

## Validation

Local validation on macOS: Release CPU build; 12 regression tests passed; all 14 added transposed-concat test cases passed on CPU; Vulkan transpose/concat shader compiled to SPIR-V with Khronos glslang 16.0.0; gfx906 table compiled and checked across types/widths/fallback modes; NVFP4 scale decoder matched upstream software arithmetic bit-for-bit for all 256 encodings. The argument-parser test could not complete its download section with the local networking/TLS configuration.

HIP compilation and AMD GPU execution/performance have not been validated in this environment. Shader compilation and CPU tests do not establish GPU correctness or speed. Before benchmarking models, run the backend checks on the machine that will execute them:

```bash
./build-gfx1030/bin/test-backend-ops -o 'MUL_MAT.*,FLASH_ATTN_EXT,GATED_DELTA_NET'
./build-vulkan/bin/test-backend-ops -o CONCAT -p 'v=1[67]'
```

For HIP, substitute the applicable build directory. Confirm that the output tests the intended GPU backend. Compare model throughput against the same upstream revision using the same model, context, batch sizes, offload, and split settings.

Source revisions and attribution are recorded in [CREDITS-AMD.md](../CREDITS-AMD.md).
