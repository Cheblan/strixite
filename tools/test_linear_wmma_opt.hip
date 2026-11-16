// Test 2-Deep Software Pipelining for linear_wmma_f16_kernel on gfx1151.
// Verifies 100% bit-identity against baseline and measures speedup.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q4.hpp"
#include "formats/q8.hpp"
#include "kernels/linear_wmma.hpp"
#include "kernels/wmma_gemm.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"
#include "runtime/q8_device.hpp"

#include <cstdio>
#include <random>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

constexpr int BN = 128;

__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = wmma_f32_to_bf16(v); }

// 1. BASELINE KERNEL (from kernels/linear_wmma.hip)
template <typename TX, typename TY, int BITS, int G, int BMT>
__global__ void __launch_bounds__(kWmmaThreads)
    linear_wmma_f16_baseline_kernel(const TX *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, TY *y,
                                    int64_t M, int64_t N, int64_t K) {
    static_assert(BITS == 4 || BITS == 8, "F16 path: Q4 / Q8 weights");
    constexpr int TM = BMT / 32;
    using WT = WmmaWCodes<BITS, G, BN>;
    __shared__ __attribute__((aligned(16))) uint16_t xs[BMT * kWmmaLd], ws[BN * kWmmaLd];
    __shared__ uint16_t sw[BN * WT::kWin], mw[BN * WT::kWin];
    __shared__ int64_t xoff[BMT];
    const int64_t m0 = (int64_t)blockIdx.y * BMT, n0 = (int64_t)blockIdx.x * BN;
    const int rows_x = (int)(M - m0 < BMT ? M - m0 : BMT), rows_w = (int)(N - n0 < BN ? N - n0 : BN);
    if (threadIdx.x < BMT) xoff[threadIdx.x] = (m0 + threadIdx.x) * K;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[TM][2] = {};
    __syncthreads();
    WmmaXTile<TX, BMT, G, false> xt;
    WT wt;
    xt.fetch(x, xoff, rows_x, 0);
    wt.fetch(q, scale, minv, n0, rows_w, K, 0);
    for (int64_t k0 = 0; k0 < K; k0 += kWmmaKC) {
        if (wt.win) {
            wt.stage_window(sw, mw);
            __syncthreads();
        }
        xt.stage_f16(xs);
        wt.stage_f16(ws, sw, mw, k0);
        __syncthreads();
        if (k0 + kWmmaKC < K) {
            xt.fetch(x, xoff, rows_x, k0 + kWmmaKC);
            wt.fetch(q, scale, minv, n0, rows_w, K, k0 + kWmmaKC);
        }
        wmma_step_f16<TM, 2>(xs, ws, wm * (BMT / 2), wn * 32, acc);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int64_t m = m0 + wm * (BMT / 2) + i * 16 + wmma_c_row(v), n = n0 + wn * 32 + j * 16 + threadIdx.x % 16;
                if (m < M && n < N) store(y, m * N + n, acc[i][j][v]);
            }
}

// 2. OPTIMIZED KERNEL: 2-Deep Software Pipelining (xa, xb and wa, wb)
template <typename TX, typename TY, int BITS, int G, int BMT>
__global__ void __launch_bounds__(kWmmaThreads)
    linear_wmma_f16_pipe2_kernel(const TX *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, TY *y,
                                 int64_t M, int64_t N, int64_t K) {
    static_assert(BITS == 4 || BITS == 8, "F16 path: Q4 / Q8 weights");
    constexpr int TM = BMT / 32;
    using WT = WmmaWCodes<BITS, G, BN>;
    __shared__ __attribute__((aligned(16))) uint16_t xs[BMT * kWmmaLd], ws[BN * kWmmaLd];
    __shared__ uint16_t sw[BN * WT::kWin], mw[BN * WT::kWin];
    __shared__ int64_t xoff[BMT];
    const int64_t m0 = (int64_t)blockIdx.y * BMT, n0 = (int64_t)blockIdx.x * BN;
    const int rows_x = (int)(M - m0 < BMT ? M - m0 : BMT), rows_w = (int)(N - n0 < BN ? N - n0 : BN);
    if (threadIdx.x < BMT) xoff[threadIdx.x] = (m0 + threadIdx.x) * K;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[TM][2] = {};
    __syncthreads();
    WmmaXTile<TX, BMT, G, false> xa, xb;
    WT wa, wb;
    xa.fetch(x, xoff, rows_x, 0);
    wa.fetch(q, scale, minv, n0, rows_w, K, 0);
    if (kWmmaKC < K) {
        xb.fetch(x, xoff, rows_x, kWmmaKC);
        wb.fetch(q, scale, minv, n0, rows_w, K, kWmmaKC);
    }
    auto step = [&](WmmaXTile<TX, BMT, G, false> &cur_x, WT &cur_w, int64_t k0) {
        if (cur_w.win) {
            cur_w.stage_window(sw, mw);
            __syncthreads();
        }
        cur_x.stage_f16(xs);
        cur_w.stage_f16(ws, sw, mw, k0);
        __syncthreads();
        if (k0 + 2 * kWmmaKC < K) {
            cur_x.fetch(x, xoff, rows_x, k0 + 2 * kWmmaKC);
            cur_w.fetch(q, scale, minv, n0, rows_w, K, k0 + 2 * kWmmaKC);
        }
        wmma_step_f16<TM, 2>(xs, ws, wm * (BMT / 2), wn * 32, acc);
        __syncthreads();
    };
    for (int64_t k0 = 0; k0 < K; k0 += 2 * kWmmaKC) {
        step(xa, wa, k0);
        if (k0 + kWmmaKC < K) step(xb, wb, k0 + kWmmaKC);
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int64_t m = m0 + wm * (BMT / 2) + i * 16 + wmma_c_row(v), n = n0 + wn * 32 + j * 16 + threadIdx.x % 16;
                if (m < M && n < N) store(y, m * N + n, acc[i][j][v]);
            }
}

void test_shape_q8(int64_t M, int64_t N, int64_t K, const char *name) {
    constexpr int G = 64;
    std::printf("================================================================================\n");
    std::printf("Testing %s: [%ld x %ld x %ld] (Q8G64, BF16)\n", name, (long)M, (long)N, (long)K);
    std::printf("================================================================================\n");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w((size_t)(N * K));
    for (float &v : w) v = dist(rng);
    Q8Weight qw = quantize_q8(w.data(), N, K, G);
    Q8Device dw = Q8Device::upload(qw, name);

    std::vector<uint16_t> host_x((size_t)(M * K));
    for (size_t i = 0; i < host_x.size(); ++i) host_x[i] = 0x3c00;  // 1.0 in BF16
    DeviceBuffer<uint16_t> d_x = DeviceBuffer<uint16_t>::from_host(host_x, "x");

    DeviceBuffer<uint16_t> d_y_base((size_t)(M * N), "y_base");
    DeviceBuffer<uint16_t> d_y_pipe2((size_t)(M * N), "y_pipe2");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    constexpr int BMT = 128;
    const int64_t col_blocks = (N + BN - 1) / BN;
    dim3 grid((unsigned)col_blocks, (unsigned)((M + BMT - 1) / BMT)), block(kWmmaThreads);

    // Run both
    linear_wmma_f16_baseline_kernel<uint16_t, uint16_t, 8, G, BMT><<<grid, block, 0, stream>>>(
        d_x.get(), dw.view().q, dw.view().scale, dw.view().minv, d_y_base.get(), M, N, K);
    linear_wmma_f16_pipe2_kernel<uint16_t, uint16_t, 8, G, BMT><<<grid, block, 0, stream>>>(
        d_x.get(), dw.view().q, dw.view().scale, dw.view().minv, d_y_pipe2.get(), M, N, K);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync");

    // Verify 100% Bit-Identity
    std::vector<uint16_t> h_base((size_t)(M * N)), h_pipe2((size_t)(M * N));
    STRIX_HIP_CHECK(hipMemcpy(h_base.data(), d_y_base.get(), h_base.size() * 2, hipMemcpyDeviceToHost), "cpy");
    STRIX_HIP_CHECK(hipMemcpy(h_pipe2.data(), d_y_pipe2.get(), h_pipe2.size() * 2, hipMemcpyDeviceToHost), "cpy");

    int64_t diffs = 0;
    for (size_t i = 0; i < h_base.size(); ++i) {
        if (h_base[i] != h_pipe2[i]) ++diffs;
    }
    std::printf("Verification: %ld diffs / %zu elements (%.6f%%)\n",
                (long)diffs, h_base.size(), (double)diffs / h_base.size() * 100.0);
    if (diffs == 0) {
        std::printf(">>> 100.000%% BIT-EXACT MATCH! <<<\n");
    } else {
        std::printf(">>> ERROR: MISMATCH DETECTED! <<<\n");
    }

    auto bench = [&](auto fn, const char *label, int iters = 25) -> float {
        for (int i = 0; i < 3; ++i) fn();
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync");
        float total_ms = 0.0f;
        for (int i = 0; i < iters; ++i) {
            STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
            fn();
            STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
            STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
            float ms = 0.0f;
            STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
            total_ms += ms;
        }
        float avg_ms = total_ms / iters;
        double tflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e12;
        std::printf("Benchmark: %-25s -> %7.3f ms  (%.2f TFLOPS)\n", label, avg_ms, tflops);
        return avg_ms;
    };

    float t_base = bench([&]() {
        linear_wmma_f16_baseline_kernel<uint16_t, uint16_t, 8, G, BMT><<<grid, block, 0, stream>>>(
            d_x.get(), dw.view().q, dw.view().scale, dw.view().minv, d_y_base.get(), M, N, K);
    }, "Baseline");

    float t_pipe2 = bench([&]() {
        linear_wmma_f16_pipe2_kernel<uint16_t, uint16_t, 8, G, BMT><<<grid, block, 0, stream>>>(
            d_x.get(), dw.view().q, dw.view().scale, dw.view().minv, d_y_pipe2.get(), M, N, K);
    }, "Pipe2 (2-Deep)");

    float speedup = (t_base - t_pipe2) / t_base * 100.0f;
    std::printf("Speedup: %+.2f%% (%7.3f ms -> %7.3f ms)\n\n", speedup, t_base, t_pipe2);

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
}

void test_shape_q4(int64_t M, int64_t N, int64_t K, const char *name) {
    constexpr int G = 128;
    std::printf("================================================================================\n");
    std::printf("Testing %s: [%ld x %ld x %ld] (Q4G128, BF16)\n", name, (long)M, (long)N, (long)K);
    std::printf("================================================================================\n");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w((size_t)(N * K));
    for (float &v : w) v = dist(rng);
    Q4Weight qw = quantize_q4(w.data(), N, K, G);
    Q4Device dw = Q4Device::upload(qw, name);

    std::vector<uint16_t> host_x((size_t)(M * K));
    for (size_t i = 0; i < host_x.size(); ++i) host_x[i] = 0x3c00;  // 1.0 in BF16
    DeviceBuffer<uint16_t> d_x = DeviceBuffer<uint16_t>::from_host(host_x, "x");

    DeviceBuffer<uint16_t> d_y_base((size_t)(M * N), "y_base");
    DeviceBuffer<uint16_t> d_y_pipe2((size_t)(M * N), "y_pipe2");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    constexpr int BMT = 128;
    const int64_t col_blocks = (N + BN - 1) / BN;
    dim3 grid((unsigned)col_blocks, (unsigned)((M + BMT - 1) / BMT)), block(kWmmaThreads);

    // Run both
    linear_wmma_f16_baseline_kernel<uint16_t, uint16_t, 4, G, BMT><<<grid, block, 0, stream>>>(
        d_x.get(), dw.view().q, dw.view().scale, dw.view().minv, d_y_base.get(), M, N, K);
    linear_wmma_f16_pipe2_kernel<uint16_t, uint16_t, 4, G, BMT><<<grid, block, 0, stream>>>(
        d_x.get(), dw.view().q, dw.view().scale, dw.view().minv, d_y_pipe2.get(), M, N, K);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync");

    // Verify 100% Bit-Identity
    std::vector<uint16_t> h_base((size_t)(M * N)), h_pipe2((size_t)(M * N));
    STRIX_HIP_CHECK(hipMemcpy(h_base.data(), d_y_base.get(), h_base.size() * 2, hipMemcpyDeviceToHost), "cpy");
    STRIX_HIP_CHECK(hipMemcpy(h_pipe2.data(), d_y_pipe2.get(), h_pipe2.size() * 2, hipMemcpyDeviceToHost), "cpy");

    int64_t diffs = 0;
    for (size_t i = 0; i < h_base.size(); ++i) {
        if (h_base[i] != h_pipe2[i]) ++diffs;
    }
    std::printf("Verification: %ld diffs / %zu elements (%.6f%%)\n",
                (long)diffs, h_base.size(), (double)diffs / h_base.size() * 100.0);
    if (diffs == 0) {
        std::printf(">>> 100.000%% BIT-EXACT MATCH! <<<\n");
    } else {
        std::printf(">>> ERROR: MISMATCH DETECTED! <<<\n");
    }

    auto bench = [&](auto fn, const char *label, int iters = 25) -> float {
        for (int i = 0; i < 3; ++i) fn();
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync");
        float total_ms = 0.0f;
        for (int i = 0; i < iters; ++i) {
            STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
            fn();
            STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
            STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
            float ms = 0.0f;
            STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
            total_ms += ms;
        }
        float avg_ms = total_ms / iters;
        double tflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e12;
        std::printf("Benchmark: %-25s -> %7.3f ms  (%.2f TFLOPS)\n", label, avg_ms, tflops);
        return avg_ms;
    };

    float t_base = bench([&]() {
        linear_wmma_f16_baseline_kernel<uint16_t, uint16_t, 4, G, BMT><<<grid, block, 0, stream>>>(
            d_x.get(), dw.view().q, dw.view().scale, dw.view().minv, d_y_base.get(), M, N, K);
    }, "Baseline");

    float t_pipe2 = bench([&]() {
        linear_wmma_f16_pipe2_kernel<uint16_t, uint16_t, 4, G, BMT><<<grid, block, 0, stream>>>(
            d_x.get(), dw.view().q, dw.view().scale, dw.view().minv, d_y_pipe2.get(), M, N, K);
    }, "Pipe2 (2-Deep)");

    float speedup = (t_base - t_pipe2) / t_base * 100.0f;
    std::printf("Speedup: %+.2f%% (%7.3f ms -> %7.3f ms)\n\n", speedup, t_base, t_pipe2);

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
}

}  // namespace

int main() {
    test_shape_q8(8192, 2560, 6144, "gdn_out / attn_o");
    test_shape_q8(8192, 13952, 2560, "attn_qkv");
    test_shape_q4(8192, 16480, 2560, "gdn_in");
    return 0;
}
