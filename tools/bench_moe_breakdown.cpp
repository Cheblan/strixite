// Detailed breakdown and sweep of MoE kernel tile configurations on gfx1151.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q4.hpp"
#include "kernels/linear_q4.hpp"
#include "kernels/moe_grouped.hpp"
#include "kernels/moe_router.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

Q4Device make_random_q4(int64_t N, int64_t K, int64_t G, std::mt19937 &rng, const std::string &name) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w((size_t)(N * K));
    for (float &v : w) v = dist(rng);
    Q4Weight qw = quantize_q4(w.data(), N, K, G);
    return Q4Device::upload(qw, name);
}

void run_breakdown(int64_t T, int iters = 20) {
    const int64_t d = 2560;
    const int64_t inter = 640;
    const int64_t experts = 512;
    const int64_t top_k = 10;
    const int64_t A = top_k;
    const int64_t G = 128;
    const Act act = Act::BF16;

    std::mt19937 rng(42);
    Q4Device gate_up = make_random_q4(experts * 2 * inter, d, G, rng, "gate_up");
    Q4Device down = make_random_q4(experts * d, inter, G, rng, "down");

    std::vector<uint16_t> host_u((size_t)(T * d), 0x3c00);
    DeviceBuffer<uint16_t> u = DeviceBuffer<uint16_t>::from_host(host_u, "u");

    std::vector<int32_t> host_ids((size_t)(T * A));
    std::vector<float> host_coef((size_t)(T * A), 0.1f);
    std::uniform_int_distribution<int32_t> exp_dist(0, (int32_t)(experts - 1));
    for (int64_t i = 0; i < T * A; ++i) host_ids[i] = exp_dist(rng);

    DeviceBuffer<int32_t> route_ids = DeviceBuffer<int32_t>::from_host(host_ids, "route_ids");
    DeviceBuffer<float> route_coef = DeviceBuffer<float>::from_host(host_coef, "route_coef");

    size_t ws_bytes = moe_route_workspace_bytes(T, A, experts);
    DeviceBuffer<uint8_t> group_ws(ws_bytes, "group_ws");
    DeviceBuffer<float> group_partial((size_t)(T * A * d), "group_partial");
    DeviceBuffer<uint32_t> err(3, "err");
    STRIX_HIP_CHECK(hipMemset(err.get(), 0, 12), "zero err");

    DeviceBuffer<uint8_t> hh(2 * (size_t)(T * A * inter), "hh");
    DeviceBuffer<uint8_t> y(2 * (size_t)(T * d), "y");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    auto time_fn = [&](auto fn) {
        // warmup
        for (int i = 0; i < 2; ++i) fn();
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync");
        float total = 0.0f;
        for (int i = 0; i < iters; ++i) {
            STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
            fn();
            STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
            STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
            float ms = 0.0f;
            STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
            total += ms;
        }
        return total / iters;
    };

    std::printf("--- Benchmark for Chunk T = %ld tokens ---\n", (long)T);

    // 1. Grouping
    float t_group = time_fn([&] {
        moe_group_routes(route_ids.get(), T, A, experts, group_ws.get(), ws_bytes, err.get(), stream);
    });
    std::printf("  moe_group_routes               : %6.3f ms\n", t_group);

    // 2. Gather SwiGLU modes
    moe_group_routes(route_ids.get(), T, A, experts, group_ws.get(), ws_bytes, err.get(), stream);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync");

    float t_gather_narrow = time_fn([&] {
        linear_q4_experts_gather_grouped_swiglu(u.get(), gate_up.view(), experts, group_ws.get(), ws_bytes,
                                                T, A, hh.get(), act, MoeMath::WmmaF16Narrow, stream, true);
    });
    std::printf("  gather_grouped_swiglu (Narrow) : %6.3f ms\n", t_gather_narrow);

    float t_gather_wide = time_fn([&] {
        linear_q4_experts_gather_grouped_swiglu(u.get(), gate_up.view(), experts, group_ws.get(), ws_bytes,
                                                T, A, hh.get(), act, MoeMath::WmmaF16Wide, stream, true);
    });
    std::printf("  gather_grouped_swiglu (Wide)   : %6.3f ms\n", t_gather_wide);

    float t_gather_wide256 = time_fn([&] {
        linear_q4_experts_gather_grouped_swiglu(u.get(), gate_up.view(), experts, group_ws.get(), ws_bytes,
                                                T, A, hh.get(), act, MoeMath::WmmaF16Wide256, stream, true);
    });
    std::printf("  gather_grouped_swiglu (Wide256): %6.3f ms\n", t_gather_wide256);

    // 3. Combine modes
    float t_comb_narrow = time_fn([&] {
        linear_q4_experts_combine_grouped(hh.get(), down.view(), experts, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, group_partial.get(), T, A, y.get(), act,
                                          MoeMath::WmmaF16Narrow, stream, nullptr, true);
    });
    std::printf("  combine_grouped (Narrow)       : %6.3f ms\n", t_comb_narrow);

    float t_comb_wide = time_fn([&] {
        linear_q4_experts_combine_grouped(hh.get(), down.view(), experts, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, group_partial.get(), T, A, y.get(), act,
                                          MoeMath::WmmaF16Wide, stream, nullptr, true);
    });
    std::printf("  combine_grouped (Wide)         : %6.3f ms\n", t_comb_wide);

    float t_comb_wide256 = time_fn([&] {
        linear_q4_experts_combine_grouped(hh.get(), down.view(), experts, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, group_partial.get(), T, A, y.get(), act,
                                          MoeMath::WmmaF16Wide256, stream, nullptr, true);
    });
    std::printf("  combine_grouped (Wide256)      : %6.3f ms\n\n", t_comb_wide256);

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
}

}  // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("   MOE ROUTING & TILING BREAKDOWN BENCHMARK ON GFX1151 (STRIX HALO)            \n");
    std::printf("================================================================================\n\n");

    run_breakdown(2048, 20);
    run_breakdown(8192, 20);

    std::printf("================================================================================\n");
    return 0;
}
