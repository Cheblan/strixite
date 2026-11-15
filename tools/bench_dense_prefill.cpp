// Standalone benchmark for Dense Projections on gfx1151.
// Measures exact execution time of every dense projection at M = 8192.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q8.hpp"
#include "kernels/linear.hpp"
#include "kernels/linear_wmma.hpp"
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

void bench_shape(const char *name, int64_t N, int64_t K, int64_t G, int64_t M, int iters = 20) {
    std::mt19937 rng(1234);
    Q8Device w = make_random_q8(N, K, G, rng, name);

    std::vector<uint16_t> host_x((size_t)(M * K), 0x3c00); // 1.0 in BF16
    DeviceBuffer<uint16_t> x = DeviceBuffer<uint16_t>::from_host(host_x, "x");
    DeviceBuffer<uint16_t> y((size_t)(M * N), "y");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    // Warmup
    for (int i = 0; i < 3; ++i) {
        linear_q8_wmma(x.get(), w.view(), y.get(), M, Act::BF16, Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "warmup sync");

    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
        linear_q8_wmma(x.get(), w.view(), y.get(), M, Act::BF16, Act::BF16, stream);
        STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
        STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
        float ms = 0.0f;
        STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
        total_ms += ms;
    }
    float avg_ms = total_ms / iters;
    double gflops = 2.0 * (double)M * (double)N * (double)K / 1e9;
    double tflops_s = gflops / avg_ms;

    std::printf("%-20s [N=%5ld, K=%5ld, G=%3ld] M=%4ld:  %6.3f ms  (%5.1f TFLOPS)\n",
                name, (long)N, (long)K, (long)G, (long)M, avg_ms, tflops_s);

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
}

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("   DENSE PROJECTIONS BENCHMARK ON GFX1151 (STRIX HALO RDNA 3.5)                 \n");
    std::printf("================================================================================\n\n");

    const int64_t M = 8192;
    bench_shape("attention_qkv", 13312, 2560, 64, M);
    bench_shape("attention_o",    2560, 6144, 64, M);
    bench_shape("gdn_out",        2560, 6144, 64, M);
    bench_shape("shared_gate_up", 1280, 2560, 64, M);
    bench_shape("shared_down",    2560,  640, 64, M);
    bench_shape("hc_mix",         2560, 10240, 64, M);

    std::printf("================================================================================\n");
    return 0;
}
