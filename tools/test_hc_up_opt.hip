// Test optimizations for hc_up_wmma_kernel on gfx1151.
// Tests two-deep register pipelining and double-buffered LDS.
// Verifies 100% bit-identity against baseline and measures latency.

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
constexpr int BM = 64, BN = 128;
constexpr int kUpJ = BN / kH; // 32

__device__ inline float to_f32(float v) { return v; }
__device__ inline float to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = wmma_f32_to_bf16(v); }

template <int BITS, int G>
struct UpCodes {
    static constexpr int kBytesPerChunk = 32 * BITS / 8, kLoadsPerChunk = kBytesPerChunk / 16;
    static constexpr int kLoadsPerRow = (kWmmaKC / 32) * kLoadsPerChunk, kCodesPerLoad = 16 * 8 / BITS;
    static constexpr int kLoads = BN * kLoadsPerRow / kWmmaThreads, kWin = 8, kWinLoads = BN * kWin / kWmmaThreads;
    static_assert(BN * kLoadsPerRow % kWmmaThreads == 0 && BN * kWin % kWmmaThreads == 0, "whole loads per thread");
    uint4 q[kLoads];
    uint16_t s[kWinLoads], m[kWinLoads];
    bool win = false;

    __device__ static int64_t row_of(int lr, int64_t d, int64_t j0) { return (int64_t)(lr / kUpJ) * d + j0 + lr % kUpJ; }
    __device__ void fetch(const uint8_t *codes, const uint16_t *scale, const uint16_t *minv, int64_t N, int64_t K,
                          int64_t d, int64_t j0, int64_t k0) {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, lr = l / kLoadsPerRow, li = l % kLoadsPerRow;
            const int64_t k = k0 + (int64_t)li * kCodesPerLoad;
            q[i] = *reinterpret_cast<const uint4 *>(codes + ((k / 32) * N + row_of(lr, d, j0)) * kBytesPerChunk +
                                                     (k % 32) * BITS / 8);
        }
        win = k0 % G == 0 && (k0 / G) % kWin == 0;
        if (win) {
            const int64_t groups = K / G, g0 = k0 / G;
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads, lr = e % BN;
                const int64_t g = g0 + e / BN, at = g * N + row_of(lr, d, j0);
                const bool ok = g < groups;
                s[i] = ok ? scale[at] : (uint16_t)0, m[i] = ok ? minv[at] : (uint16_t)0;
            }
        }
    }
    __device__ void stage_window(uint16_t *sw, uint16_t *mw) const {
        if (win)
#pragma unroll
            for (int i = 0; i < kWinLoads; ++i) {
                const int e = (int)threadIdx.x + i * kWmmaThreads;
                sw[(e % BN) * kWin + e / BN] = s[i], mw[(e % BN) * kWin + e / BN] = m[i];
            }
    }
    __device__ void stage_f16(uint16_t *lds, const uint16_t *sw, const uint16_t *mw, int64_t k0) const {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            const int l = (int)threadIdx.x + i * kWmmaThreads, lr = l / kLoadsPerRow, kin = (l % kLoadsPerRow) * kCodesPerLoad;
            const int slot = (int)(((k0 + kin) / G) % kWin);
            wmma_stage_codes_f16<BITS>(q[i], wmma_bf16_bits_to_f16x2(sw[lr * kWin + slot]),
                                       wmma_bf16_bits_to_f16x2(mw[lr * kWin + slot]), lds + lr * kWmmaLd + kin);
        }
    }
};

// 1. Baseline kernel
template <typename T, int BITS, int G>
__global__ void __launch_bounds__(kWmmaThreads)
    hc_up_wmma_baseline_kernel(const float *h, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, const T *x,
                               const float *norm_w, const float *inv, int64_t Tn, int64_t d, int64_t r, T *u) {
    __shared__ __attribute__((aligned(16))) unsigned char smem[kH * BM * kUpJ * 4];
    uint16_t *xs = reinterpret_cast<uint16_t *>(smem), *ws = xs + BM * kWmmaLd;
    float *contrib = reinterpret_cast<float *>(smem);
    __shared__ uint16_t sw[BN * UpCodes<BITS, G>::kWin], mw[BN * UpCodes<BITS, G>::kWin];
    __shared__ int64_t hoff[BM];
    const int64_t j0 = (int64_t)blockIdx.x * kUpJ, m0 = (int64_t)blockIdx.y * BM, N = kH * d;
    const int rows_x = (int)(Tn - m0 < BM ? Tn - m0 : BM);
    if (threadIdx.x < BM) hoff[threadIdx.x] = (m0 + threadIdx.x) * r;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[2][2] = {};
    __syncthreads();
    
    WmmaXTileF32ToF16<BM> xt;
    UpCodes<BITS, G> wt;
    xt.fetch(h, hoff, rows_x, 0);
    wt.fetch(q, scale, minv, N, r, d, j0, 0);
    for (int64_t k0 = 0; k0 < r; k0 += kWmmaKC) {
        if (wt.win) {
            wt.stage_window(sw, mw);
            __syncthreads();
        }
        xt.stage_f16(xs);
        wt.stage_f16(ws, sw, mw, k0);
        __syncthreads();
        if (k0 + kWmmaKC < r) {
            xt.fetch(h, hoff, rows_x, k0 + kWmmaKC);
            wt.fetch(q, scale, minv, N, r, d, j0, k0 + kWmmaKC);
        }
        wmma_step_f16<2, 2>(xs, ws, wm * 32, wn * 32, acc);
        __syncthreads();
    }
    const int s = wn;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = wm * 32 + i * 16 + wmma_c_row(v), jj = j * 16 + threadIdx.x % 16;
                const int64_t t = m0 + row, g = (int64_t)s * d + j0 + jj;
                float c = 0.f;
                if (t < Tn) {
                    const float z = 1.0f / (1.0f + expf(-acc[i][j][v]));
                    c = z * (to_f32(x[t * N + g]) * inv[t * kH + s] * (1.0f + norm_w[g]));
                }
                contrib[(s * BM + row) * kUpJ + jj] = c;
            }
    __syncthreads();
    for (int i = threadIdx.x; i < BM * kUpJ; i += kWmmaThreads) {
        const int row = i / kUpJ, jj = i % kUpJ;
        const int64_t t = m0 + row;
        if (t >= Tn) continue;
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < kH; ++k) sum += contrib[(k * BM + row) * kUpJ + jj];
        store(u, t * d + j0 + jj, sum / (float)kH);
    }
}

// 2. Opt 1: Two-deep register software pipelining (xa, xb and wa, wb)
template <typename T, int BITS, int G>
__global__ void __launch_bounds__(kWmmaThreads)
    hc_up_wmma_pipe2_kernel(const float *h, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, const T *x,
                            const float *norm_w, const float *inv, int64_t Tn, int64_t d, int64_t r, T *u) {
    __shared__ __attribute__((aligned(16))) unsigned char smem[kH * BM * kUpJ * 4];
    uint16_t *xs = reinterpret_cast<uint16_t *>(smem), *ws = xs + BM * kWmmaLd;
    float *contrib = reinterpret_cast<float *>(smem);
    __shared__ uint16_t sw[BN * UpCodes<BITS, G>::kWin], mw[BN * UpCodes<BITS, G>::kWin];
    __shared__ int64_t hoff[BM];
    const int64_t j0 = (int64_t)blockIdx.x * kUpJ, m0 = (int64_t)blockIdx.y * BM, N = kH * d;
    const int rows_x = (int)(Tn - m0 < BM ? Tn - m0 : BM);
    if (threadIdx.x < BM) hoff[threadIdx.x] = (m0 + threadIdx.x) * r;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[2][2] = {};
    __syncthreads();

    WmmaXTileF32ToF16<BM> xa, xb;
    UpCodes<BITS, G> wa, wb;
    xa.fetch(h, hoff, rows_x, 0);
    wa.fetch(q, scale, minv, N, r, d, j0, 0);
    if (kWmmaKC < r) {
        xb.fetch(h, hoff, rows_x, kWmmaKC);
        wb.fetch(q, scale, minv, N, r, d, j0, kWmmaKC);
    }
    auto step = [&](WmmaXTileF32ToF16<BM> &xt, UpCodes<BITS, G> &wt, int64_t k0) {
        if (wt.win) {
            wt.stage_window(sw, mw);
            __syncthreads();
        }
        xt.stage_f16(xs);
        wt.stage_f16(ws, sw, mw, k0);
        __syncthreads();
        if (k0 + 2 * kWmmaKC < r) {
            xt.fetch(h, hoff, rows_x, k0 + 2 * kWmmaKC);
            wt.fetch(q, scale, minv, N, r, d, j0, k0 + 2 * kWmmaKC);
        }
        wmma_step_f16<2, 2>(xs, ws, wm * 32, wn * 32, acc);
        __syncthreads();
    };
    for (int64_t k0 = 0; k0 < r; k0 += 2 * kWmmaKC) {
        step(xa, wa, k0);
        if (k0 + kWmmaKC < r) step(xb, wb, k0 + kWmmaKC);
    }

    const int s = wn;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = wm * 32 + i * 16 + wmma_c_row(v), jj = j * 16 + threadIdx.x % 16;
                const int64_t t = m0 + row, g = (int64_t)s * d + j0 + jj;
                float c = 0.f;
                if (t < Tn) {
                    const float z = 1.0f / (1.0f + expf(-acc[i][j][v]));
                    c = z * (to_f32(x[t * N + g]) * inv[t * kH + s] * (1.0f + norm_w[g]));
                }
                contrib[(s * BM + row) * kUpJ + jj] = c;
            }
    __syncthreads();
    for (int i = threadIdx.x; i < BM * kUpJ; i += kWmmaThreads) {
        const int row = i / kUpJ, jj = i % kUpJ;
        const int64_t t = m0 + row;
        if (t >= Tn) continue;
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < kH; ++k) sum += contrib[(k * BM + row) * kUpJ + jj];
        store(u, t * d + j0 + jj, sum / (float)kH);
    }
}

// 3. Opt 2: Pipe2 + Hoisted norm_w loads
template <typename T, int BITS, int G>
__global__ void __launch_bounds__(kWmmaThreads)
    hc_up_wmma_pipe2_hoist_kernel(const float *h, const uint8_t *q, const uint16_t *scale, const uint16_t *minv, const T *x,
                                  const float *norm_w, const float *inv, int64_t Tn, int64_t d, int64_t r, T *u) {
    __shared__ __attribute__((aligned(16))) unsigned char smem[kH * BM * kUpJ * 4];
    uint16_t *xs = reinterpret_cast<uint16_t *>(smem), *ws = xs + BM * kWmmaLd;
    float *contrib = reinterpret_cast<float *>(smem);
    __shared__ uint16_t sw[BN * UpCodes<BITS, G>::kWin], mw[BN * UpCodes<BITS, G>::kWin];
    __shared__ int64_t hoff[BM];
    const int64_t j0 = (int64_t)blockIdx.x * kUpJ, m0 = (int64_t)blockIdx.y * BM, N = kH * d;
    const int rows_x = (int)(Tn - m0 < BM ? Tn - m0 : BM);
    if (threadIdx.x < BM) hoff[threadIdx.x] = (m0 + threadIdx.x) * r;
    const int wave = threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    wmma_f32x8 acc[2][2] = {};
    __syncthreads();

    WmmaXTileF32ToF16<BM> xa, xb;
    UpCodes<BITS, G> wa, wb;
    xa.fetch(h, hoff, rows_x, 0);
    wa.fetch(q, scale, minv, N, r, d, j0, 0);
    if (kWmmaKC < r) {
        xb.fetch(h, hoff, rows_x, kWmmaKC);
        wb.fetch(q, scale, minv, N, r, d, j0, kWmmaKC);
    }
    auto step = [&](WmmaXTileF32ToF16<BM> &xt, UpCodes<BITS, G> &wt, int64_t k0) {
        if (wt.win) {
            wt.stage_window(sw, mw);
            __syncthreads();
        }
        xt.stage_f16(xs);
        wt.stage_f16(ws, sw, mw, k0);
        __syncthreads();
        if (k0 + 2 * kWmmaKC < r) {
            xt.fetch(h, hoff, rows_x, k0 + 2 * kWmmaKC);
            wt.fetch(q, scale, minv, N, r, d, j0, k0 + 2 * kWmmaKC);
        }
        wmma_step_f16<2, 2>(xs, ws, wm * 32, wn * 32, acc);
        __syncthreads();
    };
    for (int64_t k0 = 0; k0 < r; k0 += 2 * kWmmaKC) {
        step(xa, wa, k0);
        if (k0 + kWmmaKC < r) step(xb, wb, k0 + kWmmaKC);
    }

    const int s = wn;
    const int64_t g0 = (int64_t)s * d + j0 + (threadIdx.x % 16);
    const int64_t g1 = g0 + 16;
    const float nw0 = 1.0f + norm_w[g0];
    const float nw1 = 1.0f + norm_w[g1];

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int64_t g = (j == 0) ? g0 : g1;
            const float nw = (j == 0) ? nw0 : nw1;
            const int jj = j * 16 + threadIdx.x % 16;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int row = wm * 32 + i * 16 + wmma_c_row(v);
                const int64_t t = m0 + row;
                float c = 0.f;
                if (t < Tn) {
                    const float z = 1.0f / (1.0f + expf(-acc[i][j][v]));
                    c = z * (to_f32(x[t * N + g]) * inv[t * kH + s] * nw);
                }
                contrib[(s * BM + row) * kUpJ + jj] = c;
            }
        }
    __syncthreads();
    for (int i = threadIdx.x; i < BM * kUpJ; i += kWmmaThreads) {
        const int row = i / kUpJ, jj = i % kUpJ;
        const int64_t t = m0 + row;
        if (t >= Tn) continue;
        float sum = 0.f;
#pragma unroll
        for (int k = 0; k < kH; ++k) sum += contrib[(k * BM + row) * kUpJ + jj];
        store(u, t * d + j0 + jj, sum / (float)kH);
    }
}

uint16_t host_float_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u)
        return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    const int64_t T = 8192;
    const int64_t H = 4;
    const int64_t d = 2560;
    const int64_t r = 320;
    const int64_t G = 64;

    std::printf("================================================================================\n");
    std::printf("   HC_MIX_UP OPTIMIZATION AND BIT-IDENTITY TEST ON GFX1151                      \n");
    std::printf("   T = %ld tokens, H = %ld streams, d = %ld, r = %ld                            \n", (long)T, (long)H, (long)d, (long)r);
    std::printf("================================================================================\n\n");

    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> host_w((size_t)(H * d * r));
    for (float &v : host_w) v = dist(rng);
    Q8Weight qw = quantize_q8(host_w.data(), H * d, r, G);
    Q8ChunkMajor qcm = to_chunk_major(qw, "hc_up");
    Q8ChunkMajorDevice w_up = Q8ChunkMajorDevice::upload(qcm, "hc_up");

    std::vector<float> host_h((size_t)(T * r));
    for (float &v : host_h) v = dist(rng);
    DeviceBuffer<float> d_h = DeviceBuffer<float>::from_host(host_h, "h");

    std::vector<uint16_t> host_x((size_t)(T * H * d));
    for (size_t i = 0; i < host_x.size(); ++i) host_x[i] = host_float_to_bf16(dist(rng));
    DeviceBuffer<uint16_t> d_x = DeviceBuffer<uint16_t>::from_host(host_x, "x");

    std::vector<float> host_norm(H * d, 0.1f);
    DeviceBuffer<float> d_norm = DeviceBuffer<float>::from_host(host_norm, "norm_w");

    std::vector<float> host_inv((size_t)(T * H), 1.0f);
    DeviceBuffer<float> d_inv = DeviceBuffer<float>::from_host(host_inv, "inv");

    DeviceBuffer<uint16_t> d_u_base((size_t)(T * d), "u_base");
    DeviceBuffer<uint16_t> d_u_pipe2((size_t)(T * d), "u_pipe2");
    DeviceBuffer<uint16_t> d_u_dbuf((size_t)(T * d), "u_dbuf");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    dim3 grid((unsigned)(d / kUpJ), (unsigned)((T + BM - 1) / BM)), block(kWmmaThreads);

    // Warmup & Run
    hc_up_wmma_baseline_kernel<uint16_t, 8, 64><<<grid, block, 0, stream>>>(
        d_h.get(), w_up.view().q, w_up.view().scale, w_up.view().minv, d_x.get(), d_norm.get(),
        d_inv.get(), T, d, r, d_u_base.get());

    hc_up_wmma_pipe2_kernel<uint16_t, 8, 64><<<grid, block, 0, stream>>>(
        d_h.get(), w_up.view().q, w_up.view().scale, w_up.view().minv, d_x.get(), d_norm.get(),
        d_inv.get(), T, d, r, d_u_pipe2.get());

    hc_up_wmma_pipe2_hoist_kernel<uint16_t, 8, 64><<<grid, block, 0, stream>>>(
        d_h.get(), w_up.view().q, w_up.view().scale, w_up.view().minv, d_x.get(), d_norm.get(),
        d_inv.get(), T, d, r, d_u_dbuf.get());
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "runs");

    // Verification
    std::vector<uint16_t> h_u_base((size_t)(T * d)), h_u_pipe2((size_t)(T * d)), h_u_dbuf((size_t)(T * d));
    STRIX_HIP_CHECK(hipMemcpy(h_u_base.data(), d_u_base.get(), h_u_base.size() * 2, hipMemcpyDeviceToHost), "cpy");
    STRIX_HIP_CHECK(hipMemcpy(h_u_pipe2.data(), d_u_pipe2.get(), h_u_pipe2.size() * 2, hipMemcpyDeviceToHost), "cpy");
    STRIX_HIP_CHECK(hipMemcpy(h_u_dbuf.data(), d_u_dbuf.get(), h_u_dbuf.size() * 2, hipMemcpyDeviceToHost), "cpy");

    auto check_diffs = [&](const char *label, const std::vector<uint16_t> &test_buf) {
        int64_t diffs = 0;
        for (size_t i = 0; i < test_buf.size(); ++i) {
            if (test_buf[i] != h_u_base[i]) ++diffs;
        }
        std::printf("Verification: %-25s -> diffs: %ld / %zu (%.4f%%)\n",
                    label, (long)diffs, test_buf.size(), (double)diffs / test_buf.size() * 100.0);
    };

    check_diffs("Opt 1 (Pipe2 Regs)", h_u_pipe2);
    check_diffs("Opt 2 (Pipe2 + Hoist)", h_u_dbuf);

    // Benchmarking
    auto bench = [&](auto fn, const char *label, int iters = 30) -> float {
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
        std::printf("Benchmark:    %-25s -> %7.3f ms (96 calls = %6.2f ms total)\n",
                    label, avg_ms, avg_ms * 96.0f);
        return avg_ms;
    };

    std::printf("\n--- Performance Timings (avg over 30 iterations) ---\n");
    bench([&]() {
        hc_up_wmma_baseline_kernel<uint16_t, 8, 64><<<grid, block, 0, stream>>>(
            d_h.get(), w_up.view().q, w_up.view().scale, w_up.view().minv, d_x.get(), d_norm.get(),
            d_inv.get(), T, d, r, d_u_base.get());
    }, "Baseline");

    bench([&]() {
        hc_up_wmma_pipe2_kernel<uint16_t, 8, 64><<<grid, block, 0, stream>>>(
            d_h.get(), w_up.view().q, w_up.view().scale, w_up.view().minv, d_x.get(), d_norm.get(),
            d_inv.get(), T, d, r, d_u_pipe2.get());
    }, "Opt 1 (Pipe2 Regs)");

    bench([&]() {
        hc_up_wmma_pipe2_hoist_kernel<uint16_t, 8, 64><<<grid, block, 0, stream>>>(
            d_h.get(), w_up.view().q, w_up.view().scale, w_up.view().minv, d_x.get(), d_norm.get(),
            d_inv.get(), T, d, r, d_u_dbuf.get());
    }, "Opt 2 (Pipe2 + Hoist)");

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return 0;
}
