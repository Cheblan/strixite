// Standalone GPU verification and benchmark for Side-Stream Concurrent Shared Expert in MoE Prefill.
// Verifies 100% bit-identity against upstream sequential baseline and benchmarks throughput.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q4.hpp"
#include "kernels/linear.hpp"
#include "kernels/linear_q4.hpp"
#include "kernels/moe_grouped.hpp"
#include "kernels/moe_router.hpp"
#include "kernels/swiglu.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

uint16_t float_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u)
        return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

float bf16_to_float(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

Q4Device make_random_q4(int64_t N, int64_t K, int64_t G, std::mt19937 &rng, const std::string &name) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w((size_t)(N * K));
    for (float &v : w) v = dist(rng);
    Q4Weight qw = quantize_q4(w.data(), N, K, G);
    return Q4Device::upload(qw, name);
}

struct TestResult {
    bool passed = false;
    double max_abs_diff = 0.0;
    int64_t total_elements = 0;
    int64_t diff_elements = 0;
    double time_seq_ms = 0.0;
    double time_side_ms = 0.0;
};

TestResult run_test(int64_t T, int iters = 20) {
    const int64_t d = 2560;
    const int64_t inter = 640;
    const int64_t experts = 512;
    const int64_t top_k = 10;
    const int64_t A = top_k;
    const int64_t G = 128;
    const Act act = Act::BF16;

    std::mt19937 rng(42);

    // Create random weights
    Q4Device gate_up = make_random_q4(experts * 2 * inter, d, G, rng, "gate_up");
    Q4Device down = make_random_q4(experts * d, inter, G, rng, "down");
    Q4Device shared_gate_up = make_random_q4(2 * inter, d, G, rng, "shared_gate_up");
    Q4Device shared_down = make_random_q4(d, inter, G, rng, "shared_down");

    // Inputs
    std::vector<uint16_t> host_u((size_t)(T * d));
    std::uniform_real_distribution<float> u_dist(-1.0f, 1.0f);
    for (uint16_t &v : host_u) v = float_to_bf16(u_dist(rng));
    DeviceBuffer<uint16_t> u = DeviceBuffer<uint16_t>::from_host(host_u, "u");

    // Router assignments: select random experts for each token
    std::vector<int32_t> host_ids((size_t)(T * A));
    std::vector<float> host_coef((size_t)(T * A));
    std::uniform_int_distribution<int32_t> exp_dist(0, (int32_t)(experts - 1));
    std::uniform_real_distribution<float> coef_dist(0.01f, 1.0f);
    for (int64_t t = 0; t < T; ++t) {
        float sum = 0.0f;
        for (int64_t a = 0; a < A; ++a) {
            host_ids[t * A + a] = exp_dist(rng);
            float c = coef_dist(rng);
            host_coef[t * A + a] = c;
            sum += c;
        }
        for (int64_t a = 0; a < A; ++a) host_coef[t * A + a] /= sum;
    }
    DeviceBuffer<int32_t> route_ids = DeviceBuffer<int32_t>::from_host(host_ids, "route_ids");
    DeviceBuffer<float> route_coef = DeviceBuffer<float>::from_host(host_coef, "route_coef");

    // Router gate logits for shared expert
    std::vector<float> host_gate_logits((size_t)T);
    for (float &g : host_gate_logits) g = u_dist(rng);
    DeviceBuffer<float> router_gate = DeviceBuffer<float>::from_host(host_gate_logits, "router_gate");

    // Workspaces
    size_t ws_bytes = moe_route_workspace_bytes(T, A, experts);
    DeviceBuffer<uint8_t> group_ws(ws_bytes, "group_ws");
    DeviceBuffer<float> group_partial((size_t)(T * A * d), "group_partial");
    DeviceBuffer<uint32_t> err(3, "err");
    STRIX_HIP_CHECK(hipMemset(err.get(), 0, 12), "zero err");

    // Shared expert buffers
    DeviceBuffer<uint8_t> sh_gu(2 * (size_t)(T * 2 * inter), "sh_gu");
    DeviceBuffer<uint8_t> sh_hh(2 * (size_t)(T * inter), "sh_hh");
    DeviceBuffer<uint8_t> sh_y(2 * (size_t)(T * d), "sh_y");

    // Routed intermediate buffers
    DeviceBuffer<uint8_t> hh_seq(2 * (size_t)(T * A * inter), "hh_seq");
    DeviceBuffer<uint8_t> hh_side(2 * (size_t)(T * A * inter), "hh_side");

    // Outputs
    DeviceBuffer<uint8_t> y_seq(2 * (size_t)(T * d), "y_seq");
    DeviceBuffer<uint8_t> y_side(2 * (size_t)(T * d), "y_side");

    hipStream_t stream = nullptr, side_stream = nullptr;
    hipEvent_t side_fork = nullptr, side_join = nullptr;
    hipEvent_t start_event = nullptr, stop_event = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&side_stream, hipStreamNonBlocking), "side_stream");
    STRIX_HIP_CHECK(hipEventCreateWithFlags(&side_fork, hipEventDisableTiming), "side_fork");
    STRIX_HIP_CHECK(hipEventCreateWithFlags(&side_join, hipEventDisableTiming), "side_join");
    STRIX_HIP_CHECK(hipEventCreate(&start_event), "start_event");
    STRIX_HIP_CHECK(hipEventCreate(&stop_event), "stop_event");

    const MoeSharedAdd shared_add_seq{sh_y.get(), router_gate.get(), 1, 0, err.get()};
    const MoeSharedAdd shared_add_side{sh_y.get(), router_gate.get(), 1, 0, err.get()};
    const bool h_f16 = true;

    // Helper for linear_qw
    auto run_lin = [&](const void *x, const Q4Device &w, void *y, int64_t M, hipStream_t s) {
        linear_q4(x, w.view(), y, M, act, act, s);
    };

    // -------------------------------------------------------------
    // Path A: Sequential baseline
    // -------------------------------------------------------------
    auto run_sequential = [&] {
        run_lin(u.get(), shared_gate_up, sh_gu.get(), T, stream);
        swiglu(sh_gu.get(), sh_hh.get(), T, inter, act, stream);
        run_lin(sh_hh.get(), shared_down, sh_y.get(), T, stream);

        moe_group_routes(route_ids.get(), T, A, experts, group_ws.get(), ws_bytes, err.get(), stream);
        linear_q4_experts_gather_grouped_swiglu(u.get(), gate_up.view(), experts, group_ws.get(), ws_bytes,
                                                T, A, hh_seq.get(), act, MoeMath::WmmaF16, stream, h_f16);
        linear_q4_experts_combine_grouped(hh_seq.get(), down.view(), experts, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, group_partial.get(), T, A, y_seq.get(), act,
                                          MoeMath::WmmaF16, stream, &shared_add_seq, h_f16);
    };

    // -------------------------------------------------------------
    // Path B: Side-stream concurrent
    // -------------------------------------------------------------
    auto run_sidestream = [&] {
        STRIX_HIP_CHECK(hipEventRecord(side_fork, stream), "side_fork record");
        STRIX_HIP_CHECK(hipStreamWaitEvent(side_stream, side_fork, 0), "side_stream wait fork");

        // Concurrent shared expert on side_stream
        run_lin(u.get(), shared_gate_up, sh_gu.get(), T, side_stream);
        swiglu(sh_gu.get(), sh_hh.get(), T, inter, act, side_stream);
        run_lin(sh_hh.get(), shared_down, sh_y.get(), T, side_stream);

        STRIX_HIP_CHECK(hipEventRecord(side_join, side_stream), "side_join record");

        // Main stream: route grouping + gather
        moe_group_routes(route_ids.get(), T, A, experts, group_ws.get(), ws_bytes, err.get(), stream);
        linear_q4_experts_gather_grouped_swiglu(u.get(), gate_up.view(), experts, group_ws.get(), ws_bytes,
                                                T, A, hh_side.get(), act, MoeMath::WmmaF16, stream, h_f16);

        // Wait for side_stream shared expert before combine
        STRIX_HIP_CHECK(hipStreamWaitEvent(stream, side_join, 0), "stream wait join");
        linear_q4_experts_combine_grouped(hh_side.get(), down.view(), experts, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, group_partial.get(), T, A, y_side.get(), act,
                                          MoeMath::WmmaF16, stream, &shared_add_side, h_f16);
    };

    // Warmup
    run_sequential();
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync seq");
    run_sidestream();
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync side");

    // Check Bit-Identity
    std::vector<uint16_t> host_y_seq((size_t)(T * d));
    std::vector<uint16_t> host_y_side((size_t)(T * d));
    STRIX_HIP_CHECK(hipMemcpy(host_y_seq.data(), y_seq.get(), (size_t)(T * d) * 2, hipMemcpyDeviceToHost), "cp seq");
    STRIX_HIP_CHECK(hipMemcpy(host_y_side.data(), y_side.get(), (size_t)(T * d) * 2, hipMemcpyDeviceToHost), "cp side");

    TestResult res;
    res.total_elements = T * d;
    for (int64_t i = 0; i < T * d; ++i) {
        if (host_y_seq[i] != host_y_side[i]) {
            res.diff_elements++;
            float diff = std::abs(bf16_to_float(host_y_seq[i]) - bf16_to_float(host_y_side[i]));
            res.max_abs_diff = std::max(res.max_abs_diff, (double)diff);
        }
    }
    res.passed = (res.diff_elements == 0);

    // Timing Sequential
    float total_seq_ms = 0.0f;
    for (int it = 0; it < iters; ++it) {
        STRIX_HIP_CHECK(hipEventRecord(start_event, stream), "start");
        run_sequential();
        STRIX_HIP_CHECK(hipEventRecord(stop_event, stream), "stop");
        STRIX_HIP_CHECK(hipEventSynchronize(stop_event), "sync stop");
        float ms = 0.0f;
        STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start_event, stop_event), "elapsed");
        total_seq_ms += ms;
    }
    res.time_seq_ms = total_seq_ms / iters;

    // Timing Side-stream
    float total_side_ms = 0.0f;
    for (int it = 0; it < iters; ++it) {
        STRIX_HIP_CHECK(hipEventRecord(start_event, stream), "start");
        run_sidestream();
        STRIX_HIP_CHECK(hipEventRecord(stop_event, stream), "stop");
        STRIX_HIP_CHECK(hipEventSynchronize(stop_event), "sync stop");
        float ms = 0.0f;
        STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start_event, stop_event), "elapsed");
        total_side_ms += ms;
    }
    res.time_side_ms = total_side_ms / iters;

    // Cleanup
    (void)hipStreamDestroy(stream);
    (void)hipStreamDestroy(side_stream);
    (void)hipEventDestroy(side_fork);
    (void)hipEventDestroy(side_join);
    (void)hipEventDestroy(start_event);
    (void)hipEventDestroy(stop_event);

    return res;
}

}  // namespace

int main(int argc, char **argv) {
    std::printf("================================================================================\n");
    std::printf("   MOE PREFILL SIDE-STREAM CONCURRENCY & BIT-IDENTITY VERIFICATION TEST        \n");
    std::printf("================================================================================\n\n");

    const int64_t token_counts[] = {512, 2048, 8192};

    bool all_passed = true;
    for (int64_t T : token_counts) {
        std::printf("Testing chunk T = %ld tokens...\n", (long)T);
        TestResult res = run_test(T, 20);

        double pct_match = 100.0 * (double)(res.total_elements - res.diff_elements) / (double)res.total_elements;
        std::printf("  Bit-identity: %s\n", res.passed ? "100.000% BIT-EXACT MATCH" : "FAILED");
        std::printf("    Total outputs   : %ld BF16 elements\n", (long)res.total_elements);
        std::printf("    Diffs           : %ld (match %.3f%%)\n", (long)res.diff_elements, pct_match);
        std::printf("    Max absolute err: %.6e\n", res.max_abs_diff);

        double speedup = (res.time_seq_ms - res.time_side_ms) / res.time_seq_ms * 100.0;
        double tok_s_seq = (double)T / (res.time_seq_ms / 1e3);
        double tok_s_side = (double)T / (res.time_side_ms / 1e3);

        std::printf("  Performance (MoE block latency):\n");
        std::printf("    Sequential baseline: %.3f ms (%.1f tok/s)\n", res.time_seq_ms, tok_s_seq);
        std::printf("    Side-stream concur : %.3f ms (%.1f tok/s)\n", res.time_side_ms, tok_s_side);
        std::printf("    Latency delta      : %+.3f ms (speedup: %+.2f%%)\n\n",
                    res.time_side_ms - res.time_seq_ms, speedup);

        if (!res.passed) all_passed = false;
    }

    std::printf("================================================================================\n");
    if (all_passed) {
        std::printf("OVERALL VERIFICATION: PASSED (100.000%% bit-identity across all token counts)\n");
    } else {
        std::printf("OVERALL VERIFICATION: FAILED\n");
    }
    std::printf("================================================================================\n");

    return all_passed ? 0 : 1;
}
