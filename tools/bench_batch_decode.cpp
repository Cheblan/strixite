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

    printf("Creating Sessions 0..3...\n");
    Qwen4ExpSession s0(model, capacity, chunk);
    Qwen4ExpSession s1(model, capacity, chunk);
    Qwen4ExpSession s2(model, capacity, chunk);
    Qwen4ExpSession s3(model, capacity, chunk);

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

    return 0;
}
