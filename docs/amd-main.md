# AMD integration: selection, builds, and validation

The current upstream base is `9b56ccf00d755b0bf780140f6352a67db1ac0378`. The initial combined import was reverted; the replacement changes are separate cherry-picks with source authorship and original commit references.

## What is retained

| Change | Evidence and scope | Integration notes |
| --- | --- | --- |
| Normalization loop specialization, `mxxm-t` `f6da2e696` | The source reports zero output difference, unchanged perplexity, and passing normalization tests including oversized grids. Qwen3-14B Q8_0 prefill improved about 1.4–1.5% on one/four MI50s. | Current upstream vectorized RMSNorm is preserved. Grids above 65,535 channels or samples retain the loops and synchronization. This adaptation has not been HIP-compiled locally. |
| Small Vulkan UMA readbacks, `halo-box` `db73f798f` | Present in the maintained Strix Halo fork. Reads up to 64 KiB avoid the additional staged-copy path; large uncached mappings retain it. | Clean cherry-pick. Synchronization and host-coherent checks remain intact. No specific end-to-end speedup is claimed for this patch. |
| RDNA3.0 MXFP4 decode width, Edwin Brown `d5295d36d` | Source reports matched MMVQ/MTP probes favoring four dot groups over two on gfx1100. | Clean cherry-pick, retaining the source's `RDNA3_0` compile-time gate. It does not enable this schedule on RDNA2 or Strix Halo. |
| RDNA3.0 Q8_0 decode width, Edwin Brown `f35976c38` | Source records 47/47 correctness cases on each of two RX 7900 XTs in both modes, with representative kernel improvements. No end-to-end Q8 model benchmark was available. | Kernel cherry-picked; source launcher documentation excluded. MMQ is unchanged. [Original measurements](https://github.com/edwinbrowwn/llama.cpp-rdna2/blob/f35976c38cb25e580b02572455dd875db1bde16f/docs/rdna2-rdna3-unified.md). |

These are source-reported results, not measurements of this combined branch. The normalization change applies to CUDA/HIP generally; its performance motivation was register pressure on gfx906. gfx900 and gfx1030 retain upstream dispatch and benefit only where the applicable operations execute. GPU identity alone does not imply a throughput gain.

Already included upstream:

- gfx900 dispatch: `d9df11006f6d3cc34772379bf6897f892874656f`.
- RDNA3.5 MMQ tuning: `4dd6dd4b96f32eb53a91d659bbff8b37bbd282b9`.
- BF16 fallback for hardware without native BF16 acceleration: `ad6c66839af3c5646fba8c6c2e2087a1e4e38948`.
- Vulkan IQ3_S batched matrix-vector tuning: `ba8818cbf3ad2f27f6b50e85b959ada4734f34c3`.

## What was removed or deferred

- The complete initial import (`12f90083a`) was reverted, including the gfx906 MMQ table, native gfx1030 attention/GDN changes, and direct NVFP4 scale-conversion promotion. Whole-fork benchmarks do not establish the benefit of an isolated table, and the prior promotion changed the source feature gates.
- The transpose/concat kernel from the zero-star Strix-Halo mirror was removed. A closely related implementation exists in halo-box, but its original commit also contains other opt-in changes. It has not been reintroduced as a reconstructed patch in this pass.
- halo-box's LDS stride-padding change `fd0e9b14b` was rejected for default use: its own explanation says pad 2 violates the cooperative-matrix stride alignment contract, even though selected RADV releases handle it favorably.
- The mxxm-t small-top-k launch-bounds change depends on a different top-k implementation; it was not grafted onto current upstream's implementation.
- Activation repacking, graph fusions, speculative sidecars, new quant formats, custom launchers, and topology experiments were not merged as a package. Each requires its own dependency and correctness review.

Repository popularity was considered alongside provenance, change scope, tests, and hardware measurements. It is not a correctness test. The retained Strix source is halo-box; milpster and the Strix-Halo mirror contribute no patches to the replacement set.

## Standard builds

Use separate build directories for different targets/backends. With a ROCm toolchain that supports the selected GPU:

```bash
cmake -S . -B build-gfx1030 -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-gfx1030 --config Release -j
```

Substitute `gfx900`, `gfx906`, `gfx1100`, or `gfx1151` and a matching build-directory name as appropriate. The installed ROCm stack must support that target; older Vega support cannot be supplied by llama.cpp alone.

For Strix Halo Vulkan, with Vulkan development packages and a shader compiler installed:

```bash
cmake -S . -B build-vulkan -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vulkan --config Release -j
```

Use ordinary llama.cpp runtime options. The retained changes require no custom launcher, sidecar bundle, or enabling environment variable. Model placement, expert offload, and multi-GPU splitting remain explicit choices.

## Validation

Local validation: the macOS Release CPU build passed, all 12 selected regression tests passed, and all 145 selected normalization backend cases passed on CPU, including the existing oversized-grid cases. AMD HIP compilation, Vulkan execution, GPU numerical parity, and model throughput cannot be established on this machine. Source benchmark results must not be treated as validation of the adapted normalization patch.

On the target hardware, the existing backend suite covers ordinary and oversized normalization grids:

```bash
./build-gfx1030/bin/test-backend-ops -o 'NORM,RMS_NORM,L2_NORM,ROPE,MUL_MAT.*'
```

Substitute the intended build directory and confirm that the output tests the intended GPU. For comparisons, hold the model, context, batch sizes, expert placement, and GPU split constant against the documented upstream revision.

See [source credits](../CREDITS-AMD.md) for the original commits and authors.
