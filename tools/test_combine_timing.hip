#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q4.hpp"
#include "kernels/moe_grouped.hpp"
#include "kernels/moe_router.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"

#include <cstdio>
#include <random>
#include <vector>

using namespace strix;
using namespace strix::kernels;

int main() {
    const int64_t T = 8192;
    const int64_t d = 2560;
    const int64_t inter = 640;
    const int64_t E = 512;
    const int64_t A = 10;
    const int64_t G = 128;
    const Act act = Act::BF16;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w((size_t)(E * d * inter));
    for (float &v : w) v = dist(rng);
    Q4Weight qw = quantize_q4(w.data(), E * d, inter, G);
    Q4Device down = Q4Device::upload(qw, "down");

    std::vector<uint16_t> host_h((size_t)(T * A * inter), 0x3c00);
    DeviceBuffer<uint16_t> h = DeviceBuffer<uint16_t>::from_host(host_h, "h");

    std::vector<int32_t> host_ids((size_t)(T * A));
    std::vector<float> host_coef((size_t)(T * A), 0.1f);
    std::uniform_int_distribution<int32_t> exp_dist(0, (int32_t)(E - 1));
    for (int64_t i = 0; i < T * A; ++i) host_ids[i] = exp_dist(rng);

    DeviceBuffer<int32_t> route_ids = DeviceBuffer<int32_t>::from_host(host_ids, "route_ids");
    DeviceBuffer<float> route_coef = DeviceBuffer<float>::from_host(host_coef, "route_coef");

    size_t ws_bytes = moe_route_workspace_bytes(T, A, E);
    DeviceBuffer<uint8_t> group_ws(ws_bytes, "group_ws");
    DeviceBuffer<float> partial((size_t)(T * A * d), "partial");
    DeviceBuffer<uint32_t> err(3, "err");
    STRIX_HIP_CHECK(hipMemset(err.get(), 0, 12), "zero err");

    DeviceBuffer<uint16_t> y((size_t)(T * d), "y");
    DeviceBuffer<uint16_t> shared_s((size_t)(T * d), "shared_s");
    DeviceBuffer<float> gate_logits((size_t)(T * (E + 1)), "gate_logits");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    moe_group_routes(route_ids.get(), T, A, E, group_ws.get(), ws_bytes, err.get(), stream);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync routes");

    MoeSharedAdd shared{};
    shared.s = shared_s.get();
    shared.gate_logit = gate_logits.get() + E;
    shared.gate_ld = E + 1;
    shared.gate_index = (uint32_t)E;
    shared.err = err.get();

    // Warmup
    for (int i = 0; i < 3; ++i) {
        linear_q4_experts_combine_grouped(h.get(), down.view(), E, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, partial.get(), T, A, y.get(), act,
                                          MoeMath::WmmaF16Wide, stream, &shared, true);
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync warmup");

    auto time_fn = [&](auto fn) {
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync before");
        STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
        for (int i = 0; i < 20; ++i) fn();
        STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
        STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
        float ms = 0.0f;
        STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
        return ms / 20.0f;
    };

    float t_full = time_fn([&] {
        linear_q4_experts_combine_grouped(h.get(), down.view(), E, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, partial.get(), T, A, y.get(), act,
                                          MoeMath::WmmaF16Wide, stream, &shared, true);
    });

    float t_full_noshared = time_fn([&] {
        linear_q4_experts_combine_grouped(h.get(), down.view(), E, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, partial.get(), T, A, y.get(), act,
                                          MoeMath::WmmaF16Wide, stream, nullptr, true);
    });

    setenv("STRIX_MOE_SKIP_REDUCE", "1", 1);
    float t_gemm_wide = time_fn([&] {
        linear_q4_experts_combine_grouped(h.get(), down.view(), E, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, partial.get(), T, A, y.get(), act,
                                          MoeMath::WmmaF16Wide, stream, &shared, true);
    });

    float t_gemm_wide256 = time_fn([&] {
        linear_q4_experts_combine_grouped(h.get(), down.view(), E, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, partial.get(), T, A, y.get(), act,
                                          MoeMath::WmmaF16Wide256, stream, &shared, true);
    });
    unsetenv("STRIX_MOE_SKIP_REDUCE");

    float t_full_wide256 = time_fn([&] {
        linear_q4_experts_combine_grouped(h.get(), down.view(), E, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, partial.get(), T, A, y.get(), act,
                                          MoeMath::WmmaF16Wide256, stream, &shared, true);
    });

    std::printf("Combine Grouped Full (+shared, Wide)   : %6.3f ms\n", t_full);
    std::printf("Combine Grouped GEMM only (Wide)       : %6.3f ms\n", t_gemm_wide);
    std::printf("Combine Grouped Full (+shared, Wide256): %6.3f ms\n", t_full_wide256);
    std::printf("Combine Grouped GEMM only (Wide256)    : %6.3f ms\n", t_gemm_wide256);
    std::printf("Combine Reduce only (calc)             : %6.3f ms\n", t_full - t_gemm_wide);

    return 0;
}
