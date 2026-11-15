// Test BMT = 64 vs BMT = 128 for hc_down_wmma_kernel on gfx1151.
// Verifies 100% bit-identity and measures execution time.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q8.hpp"
#include "kernels/device_info.hpp"
#include "kernels/wmma_gemm.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q8_device.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

constexpr int kH = 4;
constexpr int BN = 128;

template <typename T, int BITS, int G, int BMT>
__global__ void __launch_bounds__(kWmmaThreads)
    hc_down_wmma_templated_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                                  int64_t R, int64_t d, int64_t Tn, float *partial) {
    constexpr int TM = BMT / 32;
    using WCodes = WmmaWCodes<BITS, G, BN>;
    __shared__ __attribute__((aligned(16))) uint16_t xs[BMT * kWmmaLd], ws[BN * kWmmaLd];
    __shared__ uint16_t sw[BN * WCodes::kWin], mw[BN * WCodes::kWin];
    __shared__ int64_t xoff[BMT];
    const int64_t K = kH * d, s = blockIdx.z, m0 = (int64_t)blockIdx.y * BMT, n0 = (int64_t)blockIdx.x * BN;
    const int rows_x = (int)(Tn - m0 < BMT ? Tn - m0 : BMT), rows_w = (int)(R - n0 < BN ? R - n0 : BN);
    if (threadIdx.x < BMT) xoff[threadIdx.x] = (m0 + threadIdx.x) * K + s * d;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[TM][2] = {};
    __syncthreads();

    WmmaXTile<T, BMT, G, false> xa, xb;
    WCodes wa, wb;
    xa.fetch(x, xoff, rows_x, 0);
    wa.fetch(q, scale, minv, n0, rows_w, K, 0, s * d);
    if (kWmmaKC < d) {
        xb.fetch(x, xoff, rows_x, kWmmaKC);
        wb.fetch(q, scale, minv, n0, rows_w, K, kWmmaKC, s * d);
    }
    auto step = [&](WmmaXTile<T, BMT, G, false> &xt, WCodes &wt, int64_t k0) {
        if (wt.win) {
            wt.stage_window(sw, mw);
            __syncthreads();
        }
        xt.stage_f16(xs);
        wt.stage_f16(ws, sw, mw, k0);
        __syncthreads();
        if (k0 + 2 * kWmmaKC < d) {
            xt.fetch(x, xoff, rows_x, k0 + 2 * kWmmaKC);
            wt.fetch(q, scale, minv, n0, rows_w, K, k0 + 2 * kWmmaKC, s * d);
        }
        wmma_step_f16<TM, 2>(xs, ws, wm * (BMT / 2), wn * 32, acc);
        __syncthreads();
    };
    for (int64_t k0 = 0; k0 < d; k0 += 2 * kWmmaKC) {
        step(xa, wa, k0);
        if (k0 + kWmmaKC < d) step(xb, wb, k0 + kWmmaKC);
    }
    float *P = partial + s * Tn * R;
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int64_t m = m0 + wm * (BMT / 2) + i * 16 + wmma_c_row(v), n = n0 + wn * 32 + j * 16 + threadIdx.x % 16;
                if (m < Tn && n < R) P[m * R + n] = acc[i][j][v];
            }
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    const int64_t T = 8192, H = 4, d = 2560, r = 320, R = 324, G = 64;

    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> host_w((size_t)(R * H * d));
    for (float &v : host_w) v = dist(rng);
    Q8Weight qw = quantize_q8(host_w.data(), R, H * d, G);
    Q8Device w_down = Q8Device::upload(qw, "hc_down");

    std::vector<uint16_t> host_x((size_t)(T * H * d));
    for (size_t i = 0; i < host_x.size(); ++i) host_x[i] = 0x3c00 + (uint16_t)(i % 100);
    DeviceBuffer<uint16_t> x = DeviceBuffer<uint16_t>::from_host(host_x, "x");

    DeviceBuffer<float> p_64((size_t)(H * T * R), "p_64");
    DeviceBuffer<float> p_128((size_t)(H * T * R), "p_128");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    dim3 grid_64((unsigned)((R + BN - 1) / BN), (unsigned)((T + 64 - 1) / 64), (unsigned)H);
    dim3 grid_128((unsigned)((R + BN - 1) / BN), (unsigned)((T + 128 - 1) / 128), (unsigned)H);
    dim3 block(kWmmaThreads);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        hc_down_wmma_templated_kernel<uint16_t, 8, 64, 64><<<grid_64, block, 0, stream>>>(
            x.get(), w_down.q.get(), w_down.scale.get(), w_down.minv.get(), R, d, T, p_64.get());
        hc_down_wmma_templated_kernel<uint16_t, 8, 64, 128><<<grid_128, block, 0, stream>>>(
            x.get(), w_down.q.get(), w_down.scale.get(), w_down.minv.get(), R, d, T, p_128.get());
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "warmup sync");

    // Bit-exactness check
    std::vector<float> res_64(H * T * R), res_128(H * T * R);
    STRIX_HIP_CHECK(hipMemcpy(res_64.data(), p_64.get(), res_64.size() * sizeof(float), hipMemcpyDeviceToHost), "d2h 64");
    STRIX_HIP_CHECK(hipMemcpy(res_128.data(), p_128.get(), res_128.size() * sizeof(float), hipMemcpyDeviceToHost), "d2h 128");

    int diffs = 0;
    for (size_t i = 0; i < res_64.size(); ++i) {
        uint32_t u64, u128;
        std::memcpy(&u64, &res_64[i], 4);
        std::memcpy(&u128, &res_128[i], 4);
        if (u64 != u128) {
            if (diffs < 5) std::printf("Diff at %zu: 64=0x%08x vs 128=0x%08x\n", i, u64, u128);
            diffs++;
        }
    }
    std::printf("Bit-Exactness Check (BM=64 vs BM=128): %s (diffs = %d / %zu)\n",
                diffs == 0 ? "100.000% MATCH (0 diffs)" : "MISMATCH", diffs, res_64.size());

    // Benchmark BM = 64
    int iters = 25;
    float total_ms_64 = 0.0f;
    for (int i = 0; i < iters; ++i) {
        STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
        hc_down_wmma_templated_kernel<uint16_t, 8, 64, 64><<<grid_64, block, 0, stream>>>(
            x.get(), w_down.q.get(), w_down.scale.get(), w_down.minv.get(), R, d, T, p_64.get());
        STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
        STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
        float ms = 0.0f;
        STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
        total_ms_64 += ms;
    }
    float avg_64 = total_ms_64 / iters;

    // Benchmark BM = 128
    float total_ms_128 = 0.0f;
    for (int i = 0; i < iters; ++i) {
        STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
        hc_down_wmma_templated_kernel<uint16_t, 8, 64, 128><<<grid_128, block, 0, stream>>>(
            x.get(), w_down.q.get(), w_down.scale.get(), w_down.minv.get(), R, d, T, p_128.get());
        STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
        STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
        float ms = 0.0f;
        STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
        total_ms_128 += ms;
    }
    float avg_128 = total_ms_128 / iters;

    std::printf("Benchmark at T = %ld tokens:\n", (long)T);
    std::printf("  hc_down_wmma (BM =  64) : %6.3f ms\n", avg_64);
    std::printf("  hc_down_wmma (BM = 128) : %6.3f ms  (%+.2f%%)\n", avg_128, (avg_64 - avg_128) / avg_64 * 100.0f);

    return 0;
}
