#include "kernels/attention.hpp"
#include "kernels/kv8_math.hpp"
#include "runtime/device_buffer.hpp"
#include "common/hip_check.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <numeric>

using namespace strix;
using namespace strix::kernels;

static uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    return static_cast<uint16_t>(x >> 16);
}

static float bf16_to_f32(uint16_t b) {
    uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

void run_prefill_bench(int64_t T, int64_t pos0, hipStream_t stream) {
    const int64_t Hq = 24, Hk = 2, D = 256, BlockSize = 32;
    const int64_t W = Hk * D; // 512
    const int64_t N = pos0 + T;
    const int64_t q_stride = Hq * 2 * D;
    const float scale = 1.0f / std::sqrt((float)D);
    const AttentionShape s{T, pos0, Hq, Hk, D, q_stride, 2 * D};

    const int64_t num_blocks = (N + BlockSize - 1) / BlockSize;
    const int64_t pool_blocks = num_blocks * 3;
    std::vector<int32_t> block_table(num_blocks);
    std::mt19937 gen(42);
    std::vector<int32_t> pool_ids(pool_blocks);
    std::iota(pool_ids.begin(), pool_ids.end(), 0);
    std::shuffle(pool_ids.begin(), pool_ids.end(), gen);
    for (int64_t i = 0; i < num_blocks; ++i) block_table[i] = pool_ids[i];

    DeviceBuffer<uint16_t> d_k_flat(N * W, "k_flat");
    DeviceBuffer<uint16_t> d_v_flat(N * W, "v_flat");

    DeviceBuffer<uint16_t> d_pool_k(pool_blocks * BlockSize * W, "pool_k");
    DeviceBuffer<uint16_t> d_pool_v(pool_blocks * BlockSize * W, "pool_v");
    DeviceBuffer<int32_t> d_bt = DeviceBuffer<int32_t>::from_host(block_table, "bt");

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> h_k_stage(T * W), h_v_stage(T * W);
    for (auto &x : h_k_stage) x = f32_to_bf16(dist(gen));
    for (auto &x : h_v_stage) x = f32_to_bf16(dist(gen));
    DeviceBuffer<uint16_t> d_k_stage = DeviceBuffer<uint16_t>::from_host(h_k_stage, "k_stage");
    DeviceBuffer<uint16_t> d_v_stage = DeviceBuffer<uint16_t>::from_host(h_v_stage, "v_stage");

    std::vector<uint16_t> h_q(T * q_stride);
    for (auto &x : h_q) x = f32_to_bf16(dist(gen));
    DeviceBuffer<uint16_t> d_q = DeviceBuffer<uint16_t>::from_host(h_q, "q");

    DeviceBuffer<uint16_t> d_out_flat(T * Hq * D, "out_flat");
    DeviceBuffer<uint16_t> d_out_paged(T * Hq * D, "out_paged");
    const size_t ws_bytes = attention_workspace_bytes(s);
    DeviceBuffer<float> d_ws(ws_bytes / 4 + 1, "ws");

    PagedKvView paged_view{
        d_pool_k.get(), d_pool_v.get(), d_bt.get(),
        N, W, BlockSize
    };

    hipEvent_t start, stop;
    STRIX_HIP_CHECK(hipEventCreate(&start), "event create");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "event create");

    const int iters = (T <= 256) ? 20 : (T <= 512) ? 10 : (T <= 1024) ? 4 : (T <= 2048) ? 2 : 1;
    const int warmup = (T <= 512) ? 2 : 1;

    // --- Benchmark 1: Flat Write (hipMemcpy/direct) vs Paged Scatter ---
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        STRIX_HIP_CHECK(hipMemcpyAsync(d_k_flat.get() + pos0 * W, d_k_stage.get(), T * W * 2, hipMemcpyDeviceToDevice, stream), "flat copy k");
        STRIX_HIP_CHECK(hipMemcpyAsync(d_v_flat.get() + pos0 * W, d_v_stage.get(), T * W * 2, hipMemcpyDeviceToDevice, stream), "flat copy v");
        paged_kv_scatter(d_k_stage.get(), d_v_stage.get(), W, d_pool_k.get(), d_pool_v.get(), d_bt.get(), T, pos0, BlockSize, W, Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync warmup");

    // Benchmark Flat Copy
    STRIX_HIP_CHECK(hipEventRecord(start, stream), "rec start");
    for (int i = 0; i < iters; ++i) {
        hipMemcpyAsync(d_k_flat.get() + pos0 * W, d_k_stage.get(), T * W * 2, hipMemcpyDeviceToDevice, stream);
        hipMemcpyAsync(d_v_flat.get() + pos0 * W, d_v_stage.get(), T * W * 2, hipMemcpyDeviceToDevice, stream);
    }
    STRIX_HIP_CHECK(hipEventRecord(stop, stream), "rec stop");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync flat write");
    float t_flat_write = 0.0f;
    STRIX_HIP_CHECK(hipEventElapsedTime(&t_flat_write, start, stop), "time flat write");
    t_flat_write /= iters;

    // Benchmark Paged Scatter
    STRIX_HIP_CHECK(hipEventRecord(start, stream), "rec start");
    for (int i = 0; i < iters; ++i) {
        paged_kv_scatter(d_k_stage.get(), d_v_stage.get(), W, d_pool_k.get(), d_pool_v.get(), d_bt.get(), T, pos0, BlockSize, W, Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipEventRecord(stop, stream), "rec stop");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync paged scatter");
    float t_paged_scatter = 0.0f;
    STRIX_HIP_CHECK(hipEventElapsedTime(&t_paged_scatter, start, stop), "time paged scatter");
    t_paged_scatter /= iters;

    // --- Benchmark 2: Attention Kernel Execution ---
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        attention(d_q.get(), d_q.get() + D, d_k_flat.get(), d_v_flat.get(), N, s, scale, d_out_flat.get(), d_ws.get(), ws_bytes, Act::BF16, stream);
        attention(d_q.get(), d_q.get() + D, paged_view, s, scale, d_out_paged.get(), d_ws.get(), ws_bytes, Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync attn warmup");

    // Flat Attention
    STRIX_HIP_CHECK(hipEventRecord(start, stream), "rec start");
    for (int i = 0; i < iters; ++i) {
        attention(d_q.get(), d_q.get() + D, d_k_flat.get(), d_v_flat.get(), N, s, scale, d_out_flat.get(), d_ws.get(), ws_bytes, Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipEventRecord(stop, stream), "rec stop");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync flat attn");
    float t_flat_attn = 0.0f;
    STRIX_HIP_CHECK(hipEventElapsedTime(&t_flat_attn, start, stop), "time flat attn");
    t_flat_attn /= iters;

    // Paged Attention
    STRIX_HIP_CHECK(hipEventRecord(start, stream), "rec start");
    for (int i = 0; i < iters; ++i) {
        attention(d_q.get(), d_q.get() + D, paged_view, s, scale, d_out_paged.get(), d_ws.get(), ws_bytes, Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipEventRecord(stop, stream), "rec stop");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync paged attn");
    float t_paged_attn = 0.0f;
    STRIX_HIP_CHECK(hipEventElapsedTime(&t_paged_attn, start, stop), "time paged attn");
    t_paged_attn /= iters;

    // Verify mathematical identity
    std::vector<uint16_t> h_out_flat(T * Hq * D), h_out_paged(T * Hq * D);
    STRIX_HIP_CHECK(hipMemcpy(h_out_flat.data(), d_out_flat.get(), h_out_flat.size() * 2, hipMemcpyDeviceToHost), "read flat");
    STRIX_HIP_CHECK(hipMemcpy(h_out_paged.data(), d_out_paged.get(), h_out_paged.size() * 2, hipMemcpyDeviceToHost), "read paged");
    float max_diff = 0.0f;
    int mismatches = 0;
    for (size_t i = 0; i < h_out_flat.size(); ++i) {
        float f1 = bf16_to_f32(h_out_flat[i]);
        float f2 = bf16_to_f32(h_out_paged[i]);
        float d = std::abs(f1 - f2);
        if (d > max_diff) max_diff = d;
        if (h_out_flat[i] != h_out_paged[i]) mismatches++;
    }

    float t_total_flat = t_flat_write + t_flat_attn;
    float t_total_paged = t_paged_scatter + t_paged_attn;
    float delta_pct = (t_total_paged - t_total_flat) / t_total_flat * 100.0f;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "| T = " << std::setw(5) << T << " | "
              << std::setw(8) << t_flat_write * 1000.0f << " us | "
              << std::setw(8) << t_paged_scatter * 1000.0f << " us | "
              << std::setw(8) << t_flat_attn * 1000.0f << " us | "
              << std::setw(8) << t_paged_attn * 1000.0f << " us | "
              << std::setw(8) << t_total_flat * 1000.0f << " us | "
              << std::setw(8) << t_total_paged * 1000.0f << " us | "
              << (delta_pct >= 0 ? "+" : "") << std::setw(5) << delta_pct << "% | "
              << (mismatches == 0 && max_diff == 0.0f ? "EXACT (0 diff)" : "MISMATCH")
              << " |" << std::endl;

    STRIX_HIP_CHECK(hipEventDestroy(start), "event destroy");
    STRIX_HIP_CHECK(hipEventDestroy(stop), "event destroy");
}

int main() {
    hipStream_t stream;
    STRIX_HIP_CHECK(hipStreamCreate(&stream), "stream create");

    std::cout << "========================================================================================================================" << std::endl;
    std::cout << "                   PREFILL BENCHMARK: FLAT (MAIN) VS PAGED ATTENTION (AMD STRIX HALO GFX1151)                         " << std::endl;
    std::cout << "========================================================================================================================" << std::endl;
    std::cout << "| Prefill Length | Flat Write | Page Scatter | Flat Attn   | Paged Attn  | Total Flat  | Total Paged | Overhead | Exactness  |" << std::endl;
    std::cout << "|:---------------|:-----------|:-------------|:------------|:------------|:------------|:------------|:---------|:-----------|" << std::endl;

    run_prefill_bench(128, 0, stream);
    run_prefill_bench(256, 0, stream);
    run_prefill_bench(512, 0, stream);
    run_prefill_bench(1024, 0, stream);
    run_prefill_bench(2048, 0, stream);

    std::cout << "========================================================================================================================" << std::endl;
    STRIX_HIP_CHECK(hipStreamDestroy(stream), "stream destroy");
    return 0;
}
