#include "llama-model.h"

#include <cstdio>
#include <memory>

static int failures = 0;

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        ++failures;
    }
}

int main() {
    const float equal4[] = {1, 1, 1, 1};
    expect(llama_tensor_split_is_valid(2048, 256, equal4, 4, 0), "equal split rotation 0");
    expect(llama_tensor_split_is_valid(2048, 256, equal4, 4, 1), "equal split rotation 1");
    expect(llama_tensor_split_is_valid(2048, 256, equal4, 4, 2), "equal split rotation 2");
    expect(llama_tensor_split_is_valid(2048, 256, equal4, 4, 3), "equal split rotation 3");

    // Rotation can turn a valid unrotated ratio into a zero-width physical rank.
    const float skewed[] = {0.45f, 0.45f, 0.05f, 0.05f};
    expect(llama_tensor_split_is_valid(2048, 128, skewed, 4, 0), "skewed split rotation 0");
    expect(!llama_tensor_split_is_valid(2048, 128, skewed, 4, 2), "skewed split rotation 2 rejected");

    const float zero_ranks[] = {1, 0, 0, 0};
    expect(!llama_tensor_split_is_valid(2048, 128, zero_ranks, 4, 0), "zero-width ranks rejected");
    expect(!llama_tensor_split_is_valid(2000, 128, equal4, 4, 0), "non-divisible width rejected");
    expect(!llama_tensor_split_is_valid(2048, 128, equal4, 1, 0), "single device rejected");
    expect(!llama_tensor_split_is_valid(0, 128, equal4, 4, 0), "zero tensor width rejected");

    // Grouped DeepSeek attention must give every GPU whole output groups.
    for (auto arch : {LLM_ARCH_DEEPSEEK4, LLM_ARCH_DEEPSEEK41}) {
        auto params = llama_model_default_params();
        params.tensor_split = equal4;
        std::unique_ptr<llama_model> model(llama_model_create(arch, params));
        model->hparams.n_layer_all = 1;
        model->hparams.n_head_arr.fill(64);
        model->hparams.n_head_kv_arr.fill(1);
        model->hparams.n_embd_head_k_full = 512;
        model->hparams.dsv4_o_group_count = 8;
        model->hparams.dsv4_o_lora_rank = 1024;
        ggml_context * ctx = ggml_init({4 * ggml_tensor_overhead(), nullptr, true});
        auto add_tensor = [&](const char * name, int64_t a, int64_t b, int64_t c) {
            auto * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, a, b, c);
            ggml_set_name(t, name);
            model->tensors_by_name.emplace_back(name, t);
            return t;
        };
        auto * wo_a = add_tensor("blk.0.attn_output_a.weight", 4096, 1024, 8);
        auto * wo_b = add_tensor("blk.0.attn_output_b.weight", 8192, 5120, 1);
        auto * wq_b = add_tensor("blk.0.attn_q_b.weight", 1280, 32768, 1);
        auto * sinks = add_tensor("blk.0.attn_sinks.weight", 64, 1, 1);
        llama_meta_device_get_split_state_userdata ud{4, model.get()};
        for (auto * tensor : {wo_a, wo_b, wq_b, sinks}) {
            const auto state = llama_meta_device_get_split_state(tensor, &ud);
            expect(state.axis >= 0 && state.axis < GGML_MAX_DIMS, "attention tensor has split axis");
            for (int gpu = 0; gpu < 4; ++gpu) {
                expect(state.ne[gpu] == tensor->ne[state.axis] / 4, "four GPUs receive equal attention groups");
            }
        }
        ggml_free(ctx);
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d tensor split validation test(s) failed\n", failures);
        return 1;
    }
    std::puts("tensor split validation tests passed");
    return 0;
}