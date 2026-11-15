// Standalone benchmark for Hyper-Connection (HC) Prefill Kernels on gfx1151.
// Measures exact execution time of hc_mix_down_wmma and hc_mix_up_wmma.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q8.hpp"
#include "kernels/hc_wmma.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q8_device.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

Q8Device make_random_q8(int64_t N, int64_t K, int64_t G, std::mt19937 &rng, const std::string &name) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w((size_t)(N * K));
    for (float &v : w) v = dist(rng);
    Q8Weight qw = quantize_q8(w.data(), N, K, G);
    return Q8Device::upload(qw, name);
}

Q8ChunkMajorDevice make_random_q8_cm(int64_t N, int64_t K, int64_t G, std::mt19937 &rng, const std::string &name) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w((size_t)(N * K));
    for (float &v : w) v = dist(rng);
    Q8Weight qw = quantize_q8(w.data(), N, K, G);
    Q8ChunkMajor qcm = to_chunk_major(qw, name.c_str());
    return Q8ChunkMajorDevice::upload(qcm, name);
}

void bench_hc_prefill(int64_t T, int iters = 20) {
    std::mt19937 rng(42);
    const int64_t H = 4;
    const int64_t d = 2560;
    const int64_t r = 320;
    const int64_t R = r + H; // 324 (inject = true)
    const int64_t G = 64;
    const float eps = 1e-6f;

    // Weights
    Q8Device w_down = make_random_q8(R, H * d, G, rng, "hc_down");
    Q8ChunkMajorDevice w_up = make_random_q8_cm(H * d, r, G, rng, "hc_up");

    // Norm weight
    std::vector<float> host_norm(H * d, 0.1f);
    DeviceBuffer<float> norm_w = DeviceBuffer<float>::from_host(host_norm, "norm_w");

    // Activations
    std::vector<uint16_t> host_x((size_t)(T * H * d), 0x3c00);
    DeviceBuffer<uint16_t> x = DeviceBuffer<uint16_t>::from_host(host_x, "x");
    DeviceBuffer<uint16_t> u((size_t)(T * d), "u");
    DeviceBuffer<float> h((size_t)(T * r), "h");
    DeviceBuffer<float> w_in((size_t)(T * H), "w_in");
    DeviceBuffer<float> inv((size_t)(T * H), "inv");

    size_t ws_bytes = hc_wmma_workspace_bytes(T, H, R);
    DeviceBuffer<float> ws(ws_bytes / sizeof(float), "hc_ws");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    // Warmup
    for (int i = 0; i < 3; ++i) {
        hc_mix_down_wmma(x.get(), w_down.view(), T, H, d, r, true, eps, h.get(), w_in.get(),
                         inv.get(), ws.get(), ws_bytes, Act::BF16, stream, false);
        hc_mix_up_wmma(h.get(), w_up.view(), x.get(), norm_w.get(), inv.get(), T, H, d, u.get(),
                       Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "warmup sync");

    // Benchmark Down (with inv recomputation)
    float total_down_inv_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
        hc_mix_down_wmma(x.get(), w_down.view(), T, H, d, r, true, eps, h.get(), w_in.get(),
                         inv.get(), ws.get(), ws_bytes, Act::BF16, stream, false);
        STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
        STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
        float ms = 0.0f;
        STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
        total_down_inv_ms += ms;
    }
    float avg_down_inv_ms = total_down_inv_ms / iters;

    // Benchmark Down (inv_ready = true)
    float total_down_ready_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
        hc_mix_down_wmma(x.get(), w_down.view(), T, H, d, r, true, eps, h.get(), w_in.get(),
                         inv.get(), ws.get(), ws_bytes, Act::BF16, stream, true);
        STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
        STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
        float ms = 0.0f;
        STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
        total_down_ready_ms += ms;
    }
    float avg_down_ready_ms = total_down_ready_ms / iters;

    // Benchmark Up
    float total_up_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
        hc_mix_up_wmma(h.get(), w_up.view(), x.get(), norm_w.get(), inv.get(), T, H, d, u.get(),
                       Act::BF16, stream);
        STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
        STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
        float ms = 0.0f;
        STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
        total_up_ms += ms;
    }
    float avg_up_ms = total_up_ms / iters;

    float total_hc_ms = avg_down_ready_ms + avg_up_ms;
    float total_48l_s = (total_hc_ms * 96.0f) / 1000.0f;

    std::printf("--- T = %4ld tokens ---\n", (long)T);
    std::printf("  hc_down_wmma (calc inv)   : %6.3f ms\n", avg_down_inv_ms);
    std::printf("  hc_down_wmma (inv_ready)  : %6.3f ms\n", avg_down_ready_ms);
    std::printf("  hc_mix_up_wmma            : %6.3f ms\n", avg_up_ms);
    std::printf("  Total hc_mix (per call)   : %6.3f ms\n", total_hc_ms);
    std::printf("  Total 96 calls (48 layers): %6.3f s per chunk\n\n", total_48l_s);

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
}

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("   HYPER-CONNECTION (HC) PREFILL BREAKDOWN BENCHMARK ON GFX1151                 \n");
    std::printf("================================================================================\n\n");

    bench_hc_prefill(2048);
    bench_hc_prefill(4096);
    bench_hc_prefill(8192);

    std::printf("================================================================================\n");
    return 0;
}
