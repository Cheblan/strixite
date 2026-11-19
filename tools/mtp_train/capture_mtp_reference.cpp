#include "common/check.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/server_config.hpp"
#include "serve/log.hpp"
#include "serve/tokenizer.hpp"
#include "serve/qwen4exp_backend.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

using namespace strix;

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    std::string conf_path = "/home/cheblan/strix-server-cost-mtp.conf";
    std::string overlay_path = "/tmp/test_nonmoe_overlay.strixw";
    std::string out_path = "/tmp/mtp_reference.bin";
    std::string prompt =
        "The zero-copy Unified Memory Architecture exposes concurrent lock contention across "
        "the kernel-bypass HSA queue ringbuffers when asynchronous prefill kernels preempt "
        "speculative verification sweeps.";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) conf_path = argv[++i];
        else if (arg == "--overlay" && i + 1 < argc) overlay_path = argv[++i];
        else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (arg == "--prompt" && i + 1 < argc) prompt = argv[++i];
    }

    printf("======================================================================\n");
    printf(">>> CAPTURING MTP REFERENCE SNAPSHOT FROM STRIXITE <<<\n");
    printf("======================================================================\n");
    printf("Config:  %s\n", conf_path.c_str());
    printf("Overlay: %s\n", overlay_path.c_str());
    printf("Output:  %s\n", out_path.c_str());
    printf("Prompt:  \"%s\"\n\n", prompt.c_str());

    ServerSettings settings(strix_server_options("/models"));
    settings.load({"--config", conf_path});

    const std::string weights = settings.text("weights");
    const std::string ngram = settings.text("ngram");
    const std::string tok_path = settings.text("tokenizer");

    printf("Loading Tokenizer: %s\n", tok_path.c_str());
    Tokenizer tokenizer(tok_path);
    std::vector<int32_t> prompt_tokens = tokenizer.encode(prompt);
    printf("Prompt encoded into %zu tokens\n", prompt_tokens.size());

    PrecisionOptions prec;
    prec.gdn = parse_gdn_precision(settings.text("gdn-precision"));
    prec.attention = parse_attention_precision(settings.text("attention-precision"));

    printf("Loading Qwen4ExpModel with overlay: %s...\n", overlay_path.c_str());
    Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, false, 8388608, 1.0f, WeightPlacement::Device,
                        /*vision_tower=*/"", overlay_path);
    STRIX_CHECK(model.has_mtp(), "Model must have MTP enabled");

    const int64_t capacity = 65536;
    const int64_t chunk = 8192;
    const int64_t total_slots = 1;

    printf("Initializing Qwen4ExpBackend...\n");
    Qwen4ExpBackend backend(model, capacity, chunk, /*use_mtp=*/true, /*mtp_vocab=*/0, prec,
                            KvFormat::Q8, total_slots);

    backend.reset(0);
    printf("Executing cold prefill for %zu tokens...\n", prompt_tokens.size());
    std::vector<float> prefill_logits = backend.forward(prompt_tokens, true);

    // Greedy token from prefill
    int32_t token_0 = (int32_t)(std::max_element(prefill_logits.begin(), prefill_logits.end()) - prefill_logits.begin());
    float token_0_val = prefill_logits[token_0];
    printf("Cold prefill finished. Next token_0 = %d (logit = %.4f)\n", token_0, token_0_val);

    // Capture h_t [10240] BF16
    const size_t n4 = 4 * 2560; // 10240
    std::vector<uint16_t> h_t_bf16(n4);
    backend.session().get_mtp_prev(0, h_t_bf16.data(), n4 * sizeof(uint16_t));
    printf("Captured h_t [10240] BF16 from sq.mtp_prev\n");

    struct StepData {
        int32_t step;
        int32_t token_id;
        int32_t best_id;
        float best_v;
        float second_v;
        std::vector<uint16_t> mtp_x_bf16;
        std::vector<float> logits;
    };

    std::vector<StepData> steps;
    int32_t cur_tok = token_0;

    for (int k = 0; k <= 6; ++k) {
        printf("\n--- Running MTP Step %d with token_id=%d ---\n", k, cur_tok);
        std::vector<float> logits = backend.session().forward_mtp_seq(0, cur_tok, k);

        std::vector<uint16_t> x_bf16(n4);
        backend.session().get_mtp_x(0, x_bf16.data(), n4 * sizeof(uint16_t));

        // Find top-2
        int32_t best_id = -1, second_id = -1;
        float best_v = -1e30f, second_v = -1e30f;
        for (size_t i = 0; i < logits.size(); ++i) {
            float v = logits[i];
            if (v > best_v) {
                second_v = best_v;
                second_id = best_id;
                best_v = v;
                best_id = (int32_t)i;
            } else if (v > second_v) {
                second_v = v;
                second_id = (int32_t)i;
            }
        }

        printf("Step %d result: best_id=%d (logit=%.4f), second_id=%d (logit=%.4f), delta=%.4f\n",
               k, best_id, best_v, second_id, second_v, best_v - second_v);

        StepData sd;
        sd.step = k;
        sd.token_id = cur_tok;
        sd.best_id = best_id;
        sd.best_v = best_v;
        sd.second_v = second_v;
        sd.mtp_x_bf16 = std::move(x_bf16);
        sd.logits = std::move(logits);
        steps.push_back(std::move(sd));

        // Next step continues autoregressively with best_id
        cur_tok = best_id;
    }

    // Write binary file
    printf("\nWriting reference snapshot to: %s\n", out_path.c_str());
    std::ofstream out(out_path, std::ios::binary);
    STRIX_CHECK(out.good(), "Failed to open output file: ", out_path);

    const char magic[8] = {'M', 'T', 'P', 'R', 'E', 'F', '0', '1'};
    out.write(magic, 8);

    uint32_t n_prompt = (uint32_t)prompt_tokens.size();
    out.write(reinterpret_cast<const char *>(&n_prompt), 4);
    out.write(reinterpret_cast<const char *>(prompt_tokens.data()), n_prompt * sizeof(int32_t));

    uint32_t nh = (uint32_t)n4;
    out.write(reinterpret_cast<const char *>(&nh), 4);
    out.write(reinterpret_cast<const char *>(h_t_bf16.data()), nh * sizeof(uint16_t));

    uint32_t num_steps = (uint32_t)steps.size();
    out.write(reinterpret_cast<const char *>(&num_steps), 4);

    for (const auto &sd : steps) {
        int32_t st_idx = sd.step;
        int32_t tok_id = sd.token_id;
        int32_t b_id = sd.best_id;
        float b_val = sd.best_v;
        float s_val = sd.second_v;

        out.write(reinterpret_cast<const char *>(&st_idx), 4);
        out.write(reinterpret_cast<const char *>(&tok_id), 4);
        out.write(reinterpret_cast<const char *>(&b_id), 4);
        out.write(reinterpret_cast<const char *>(&b_val), 4);
        out.write(reinterpret_cast<const char *>(&s_val), 4);

        uint32_t nx = (uint32_t)sd.mtp_x_bf16.size();
        out.write(reinterpret_cast<const char *>(&nx), 4);
        out.write(reinterpret_cast<const char *>(sd.mtp_x_bf16.data()), nx * sizeof(uint16_t));

        uint32_t nlog = (uint32_t)sd.logits.size();
        out.write(reinterpret_cast<const char *>(&nlog), 4);
        out.write(reinterpret_cast<const char *>(sd.logits.data()), nlog * sizeof(float));
    }

    out.close();
    printf("Successfully captured reference snapshot! (%zu bytes)\n", (size_t)out.tellp());
    return 0;
}
