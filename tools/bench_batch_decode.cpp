#include "common/check.hpp"
#include "runtime/qwen4exp.hpp"
#include "serve/server_config.hpp"
#include "serve/log.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace strix;

int main(int argc, char **argv) {
    const std::string conf_path = argc > 1 ? argv[1] : "/models/Qwen-3.8-FN-strixw/strix-server.conf";
    printf("Loading config from %s\n", conf_path.c_str());
    ServerSettings settings(strix_server_options("/models"));
    settings.load({"--config", conf_path});

    const std::string weights = settings.text("weights");
    const std::string ngram = settings.text("ngram");
    const int64_t capacity = 65536;
    const int64_t chunk = 4096;

    printf("Loading Qwen4ExpModel...\n");
    Qwen4ExpModel model(weights, ngram, kernels::Act::BF16, false, 8388608, 1.0f);
    printf("Model dims: layers=%ld, ple_layer=%ld\n", (long)model.dims().layers, (long)model.dims().ple_layer);
    for (int i = 0; i < 6; ++i) {
        printf("L%d: %s\n", i, model.dims().is_attention[i] ? "ATTN" : "GDN");
    }

    printf("Creating Sessions 0..3 (with MTP)...\n");
    Qwen4ExpSession s0(model, capacity, chunk, PrefillMath::WmmaBf16, true);
    Qwen4ExpSession s1(model, capacity, chunk, PrefillMath::WmmaBf16, true);
    Qwen4ExpSession s2(model, capacity, chunk, PrefillMath::WmmaBf16, true);
    Qwen4ExpSession s3(model, capacity, chunk, PrefillMath::WmmaBf16, true);

    std::vector<int32_t> prompt = {151644, 872, 198, 2610, 525, 264, 10925, 1782, 151645, 198, 151646, 198};
    s0.forward(prompt, 1);
    s1.forward(prompt, 1);
    s2.forward(prompt, 1);
    s3.forward(prompt, 1);

    int32_t tok0 = 1234;
    int32_t tok1 = 5678;
    int32_t tok2 = 9101;
    int32_t tok3 = 1121;

    printf("\n--- Single-Token Forward Baseline (4 slots) ---\n");
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> single0 = s0.forward({tok0}, 1);
    auto t1 = std::chrono::steady_clock::now();
    std::vector<float> single1 = s1.forward({tok1}, 1);
    auto t2 = std::chrono::steady_clock::now();
    std::vector<float> single2 = s2.forward({tok2}, 1);
    auto t3 = std::chrono::steady_clock::now();
    std::vector<float> single3 = s3.forward({tok3}, 1);
    auto t4 = std::chrono::steady_clock::now();

    double ms0 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms1 = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double ms2 = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double ms3 = std::chrono::duration<double, std::milli>(t4 - t3).count();
    double ms_sum = ms0 + ms1 + ms2 + ms3;
    printf("Slots single steps: %.2f + %.2f + %.2f + %.2f = %.2f ms (%.2f tok/s aggregate)\n",
           ms0, ms1, ms2, ms3, ms_sum, 4000.0 / ms_sum);

    // Batched 4 slots:
    s0.reset(); s1.reset(); s2.reset(); s3.reset();
    s0.forward(prompt, 1); s1.forward(prompt, 1); s2.forward(prompt, 1); s3.forward(prompt, 1);

    std::vector<Qwen4ExpSession::BatchSlotStep> batch4 = {
        {&s0, tok0}, {&s1, tok1}, {&s2, tok2}, {&s3, tok3}
    };
    std::vector<std::vector<float>> batch4_logits;

    // Warmup
    Qwen4ExpSession::forward_batch_decode(batch4, batch4_logits);

    // Reset and measured run:
    s0.reset(); s1.reset(); s2.reset(); s3.reset();
    s0.forward(prompt, 1); s1.forward(prompt, 1); s2.forward(prompt, 1); s3.forward(prompt, 1);

    auto tb0 = std::chrono::steady_clock::now();
    Qwen4ExpSession::forward_batch_decode(batch4, batch4_logits);
    auto tb1 = std::chrono::steady_clock::now();

    double ms_batch4 = std::chrono::duration<double, std::milli>(tb1 - tb0).count();
    printf("\n--- Batched Decode Forward (4 slots) ---\n");
    printf("Batched 4 slots step: %.2f ms (%.2f tok/s aggregate)\n", ms_batch4, 4000.0 / ms_batch4);
    printf("Speedup over unbatched: %.2fx\n", ms_sum / ms_batch4);

    float max_diffs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const std::vector<float> *singles[4] = {&single0, &single1, &single2, &single3};
    for (int k = 0; k < 4; ++k) {
        for (size_t i = 0; i < singles[k]->size(); ++i) {
            float d = std::abs((*singles[k])[i] - batch4_logits[k][i]);
            if (d > max_diffs[k]) max_diffs[k] = d;
        }
        printf("Slot %d max logit difference (batch vs single): %g\n", k, max_diffs[k]);
    }

    // --- Batched MTP Speculative Decode Benchmark ---
    printf("\n--- Batched MTP Speculative Decode (4 slots) ---\n");
    s0.reset(); s1.reset(); s2.reset(); s3.reset();
    s0.forward(prompt, 1); s1.forward(prompt, 1); s2.forward(prompt, 1); s3.forward(prompt, 1);

    std::vector<Qwen4ExpSession *> sessions = {&s0, &s1, &s2, &s3};
    std::vector<int32_t> cur_tokens = {tok0, tok1, tok2, tok3};

    // Warmup:
    Qwen4ExpSession::step_batch_mtp(sessions, cur_tokens, 2, 0.15f);

    // Benchmark 1 draft vs 2 drafts:
    for (int max_draft : {1, 2}) {
        printf("\n--- Batched MTP Speculative Decode (4 slots, max_draft=%d) ---\n", max_draft);
        s0.reset(); s1.reset(); s2.reset(); s3.reset();
        s0.forward(prompt, 1); s1.forward(prompt, 1); s2.forward(prompt, 1); s3.forward(prompt, 1);
        cur_tokens = {tok0, tok1, tok2, tok3};

        int64_t total_tokens_generated = 0;
        const int kMtpSteps = 10;
        double total_ms_draft = 0.0, total_ms_forward = 0.0;

        auto t_mtp0 = std::chrono::steady_clock::now();
        for (int step = 0; step < kMtpSteps; ++step) {
            auto td0 = std::chrono::steady_clock::now();
            std::vector<Qwen4ExpSession::BatchSlotStep> batch(4);
            for (int k = 0; k < 4; ++k) {
                batch[k].session = sessions[k];
                batch[k].token_id = cur_tokens[k];
                batch[k].tokens = {cur_tokens[k]};
                if (sessions[k]->has_mtp()) {
                    auto t0 = sessions[k]->forward_mtp_top2(cur_tokens[k], 0);
                    if (!t0.nan && t0.best_v - t0.second_v >= 0.15f) {
                        batch[k].tokens.push_back(t0.best);
                        if (max_draft >= 2) {
                            auto t1 = sessions[k]->forward_mtp_top2(t0.best, 1);
                            if (!t1.nan && t1.best_v - t1.second_v >= 0.15f) {
                                batch[k].tokens.push_back(t1.best);
                            }
                        }
                    }
                }
            }
            auto td1 = std::chrono::steady_clock::now();
            total_ms_draft += std::chrono::duration<double, std::milli>(td1 - td0).count();

            std::vector<std::vector<float>> batch_logits;
            auto tf0 = std::chrono::steady_clock::now();
            Qwen4ExpSession::forward_batch_decode(batch, batch_logits);
            auto tf1 = std::chrono::steady_clock::now();
            total_ms_forward += std::chrono::duration<double, std::milli>(tf1 - tf0).count();

            const int64_t vocab = model.dims().vocab;
            auto argmax_row = [&](const float *row) -> int32_t {
                int32_t best = 0;
                float best_v = row[0];
                for (int32_t i = 1; i < vocab; ++i) {
                    if (row[i] > best_v) { best_v = row[i]; best = i; }
                }
                return best;
            };

            for (int k = 0; k < 4; ++k) {
                Qwen4ExpSession *s = sessions[k];
                const int64_t Tk = (int64_t)batch[k].tokens.size();
                if (Tk == 1) {
                    int32_t next_tok = argmax_row(batch_logits[k].data());
                    total_tokens_generated += 1;
                    cur_tokens[k] = next_tok;
                } else {
                    int64_t kept = 1;
                    int32_t final_tok = -1;
                    for (int64_t j = 0; j < Tk - 1; ++j) {
                        int32_t sampled = argmax_row(batch_logits[k].data() + (size_t)(j * vocab));
                        if (sampled == batch[k].tokens[(size_t)(j + 1)]) {
                            total_tokens_generated += 1;
                            kept++;
                        } else {
                            final_tok = sampled;
                            break;
                        }
                    }
                    if (kept == Tk) {
                        int32_t sampled_last = argmax_row(batch_logits[k].data() + (size_t)((Tk - 1) * vocab));
                        total_tokens_generated += 1;
                        cur_tokens[k] = sampled_last;
                        s->keep_verify();
                    } else {
                        total_tokens_generated += 1;
                        cur_tokens[k] = final_tok;
                        s->keep_verify_prefix(kept);
                    }
                }
            }
        }
        auto t_mtp1 = std::chrono::steady_clock::now();
        double ms_mtp = std::chrono::duration<double, std::milli>(t_mtp1 - t_mtp0).count();
        printf("Batched MTP (%d steps): %ld tokens in %.2f ms (%.2f tok/s aggregate, %.2f tok/step)\n",
               kMtpSteps, (long)total_tokens_generated, ms_mtp,
               (double)total_tokens_generated / (ms_mtp / 1000.0),
               (double)total_tokens_generated / kMtpSteps);
        printf("  Breakdown per step: draft = %.2f ms, trunk verify = %.2f ms\n",
               total_ms_draft / kMtpSteps, total_ms_forward / kMtpSteps);
        printf("Speedup over unbatched serial: %.2fx\n",
               ((double)total_tokens_generated / (ms_mtp / 1000.0)) / (4000.0 / ms_sum));
    }

    return 0;
}
