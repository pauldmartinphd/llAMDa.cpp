# DeepSeek-V4.1-Flash-1017 and DSpark

The shared implementation is Daniel Han / Unsloth's [llama.cpp PR 30817](https://github.com/ggml-org/llama.cpp/pull/30817), cherry-picked from `b72050d03ae14bb3cb5138ecefab3f3f65157d31` with authorship preserved. It replaces the older Victor Cruz / midagedev integration and its sparse-attention adaptations. The source history retains their credits; their model implementation is superseded.

## Upstream review, 2026-10-21

The common upstream base is `9b56ccf00d755b0bf780140f6352a67db1ac0378`. PR 30817 remains open. It includes the V4.1-Flash-1017 graph, sparse attention, Engram lookup, MXFP8 gather support, model/DSpark/vision conversion, a dedicated V4.1 DSpark decoder and the softplus numerical correction. Its author reports comparisons with the official F32 reference, including a 17K-token case. Those are source-reported results, not local hardware validation.

The older [PR 28696](https://github.com/ggml-org/llama.cpp/pull/28696) remains at the previously imported revision. [PR 30657](https://github.com/ggml-org/llama.cpp/pull/30657), which adds log-SNR conditioning to other DSpark drafts, explicitly rejects the DSV4 hyper-connection backbone; it is not imported.

## Models

Use matching [Unsloth DeepSeek-V4.1-Flash-1017 GGUFs](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF). The older V4.1 GGUFs use different metadata and must not be assumed compatible. This command selects the nine-shard UD-Q4_K_XL target and the matching Q8_0 DSpark file:

```bash
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
  --include "UD-Q4_K_XL/*.gguf" "dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf" \
  --local-dir /models/DeepSeek-V4.1-Flash-1017
```

The selection is a download example, not a performance recommendation or a guarantee that GPU placement fits. Both filenames were checked against the repository listing.

## Build

Use normal llama.cpp build options; no custom launcher or HIP sidecar library is required. For a V620:

```bash
cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1030 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32
```

Use the appropriate GPU target and a separate build directory for other architectures.

## Run with routed experts in RAM

```bash
MODEL=/models/DeepSeek-V4.1-Flash-1017/UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf
DSPARK_MODEL=/models/DeepSeek-V4.1-Flash-1017/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf

build/bin/llama-server -m "$MODEL" \
  --model-draft "$DSPARK_MODEL" --spec-type draft-dspark \
  -ngl 99 --cpu-moe --split-mode layer --tensor-split 1,1,1,1 \
  --spec-draft-ngl 99 --spec-draft-cpu-moe \
  --spec-draft-n-max 8 --spec-draft-p-min 0 \
  --load-mode none --lazy-mode off -fa on -c 8192 -np 1 -t 32 \
  --host 127.0.0.1 --port 8081
```

The runtime reads the trained block size from the draft metadata and bounds the requested draft length to it. Eight is a request, not an override of the model's block size. The four-way split above is layer placement, not tensor-parallel execution. The target and draft both need room for their remaining GPU weights, KV storage and work buffers.

## Known issue and validation

The [PR review thread reports output corruption with prompt-prefix reuse](https://github.com/ggml-org/llama.cpp/pull/30817#issuecomment-6308748500), including with lazy loading disabled. Its reported workaround is to send `"cache_prompt": false` on each request. An earlier upload-race diagnosis was corrected by the reporter; no speculative synchronization patch is imported here. This remains an open issue, not a demonstrated fix in this integration.

The shared implementation passes a complete macOS CPU build, seven targeted CPU tests and the synthetic DeepSeek-V4.1 architecture test, including model serialization roundtrip. The synthetic fixture does not establish real-model long-context quality or DSpark acceptance. HIP/Vulkan compilation, actual model inference and performance on AMD hardware still require validation.
