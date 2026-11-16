// Benchmark and Bit-Identity Verification for 2-Deep Pipelined grouped_wmma_f16_wide kernel.
// Tests on AMD Ryzen AI Max+ 395 (gfx1151).

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q4.hpp"
#include "kernels/moe_grouped.hpp"
#include "kernels/moe_router.hpp"
#include "kernels/wmma_gemm.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"

#include <cstdio>
#include <random>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

constexpr int kWmmaBN = 128;
constexpr int R = 16, RW = kMoeWmmaTileRows, RX = kMoeWmmaWideRows;

__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__device__ inline void put_partial(float *p, int64_t i, float v, bool bf16) {
    if (bf16) reinterpret_cast<uint16_t *>(p)[i] = f32_to_bf16_bits(v);
    else p[i] = v;
}

struct RouteLayout {
    int64_t S, E, Tmax, WTmax, XTmax;
};

RouteLayout route_layout(int64_t M, int64_t A, int64_t E) {
    const int64_t S = M * A;
    return {S, E, S / R + E + 2, S / RW + E + 2, S / RX + E + 2};
}

// 1. BASELINE KERNEL (as in kernels/moe_grouped.hip)
template <typename T, int G, bool COMBINE, bool SWI, bool XH = false>
__global__ void __launch_bounds__(kWmmaThreads)
    grouped_wmma_f16_wide_baseline_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                                         const int32_t *ws, int64_t A, int64_t N, int64_t K, T *y, float *partial, bool pbf16) {
    static_assert(!(SWI && COMBINE), "SwiGLU fuses into the gather only");
    using WT = WmmaWCodes<4, G, kWmmaBN>;
    constexpr int TR = RX, kHalf = kWmmaBN / 2, kTileElems = (TR + kWmmaBN) * kWmmaLd;
    constexpr int kOutLd = kWmmaBN + 8;
    __shared__ __attribute__((aligned(16))) uint16_t tiles_lds[kTileElems];
    uint16_t *xs = tiles_lds, *wsm = tiles_lds + TR * kWmmaLd;
    __shared__ uint16_t sw[kWmmaBN * WT::kWin], mw[kWmmaBN * WT::kWin];
    __shared__ int64_t xoff[TR], ooff[TR];
    const int32_t S = ws[1], E = ws[2];
    const int64_t Tmax = (int64_t)S / R + E + 2, WTmax = (int64_t)S / RW + E + 2, XTmax = (int64_t)S / RX + E + 2;
    const int32_t *offsets = ws + 4, *slots = offsets + E + 2;
    const int32_t *xtile_bucket = slots + S + 2 * Tmax + 2 * WTmax, *xtile_start = xtile_bucket + XTmax;
    const int64_t tile = blockIdx.y;
    if (tile >= xtile_start[XTmax]) return;
    const int32_t bucket = xtile_bucket[tile], st = xtile_start[tile];
    const int rows = min(TR, offsets[bucket + 1] - st), nt = (rows + 15) / 16;
    constexpr int kCols = SWI ? kHalf : kWmmaBN;
    const int64_t n0 = (int64_t)blockIdx.x * kCols, Nh = SWI ? N / 2 : N;
    const int rows_w = (int)(Nh - n0 < kCols ? Nh - n0 : kCols);
    if ((int)threadIdx.x < rows) {
        const int64_t slot = slots[st + threadIdx.x];
        xoff[threadIdx.x] = (COMBINE ? slot : slot / A) * K;
        ooff[threadIdx.x] = slot * (SWI ? N / 2 : N);
    }
    __syncthreads();
    if (bucket == E) {
        for (int i = threadIdx.x; i < rows * rows_w; i += kWmmaThreads) {
            const int r = i / rows_w, c = i % rows_w;
            if (COMBINE) put_partial(partial, ooff[r] + n0 + c, __int_as_float(0x7fc00000), pbf16);
        }
        return;
    }
    const int wave = (int)threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    const int64_t wbase = (int64_t)bucket * N + n0;
    const bool vec_out = N % 8 == 0 && reinterpret_cast<uintptr_t>(partial) % 16 == 0;
    const auto row_of = [&](int r) -> int64_t {
        const int c = SWI ? r % kHalf : r;
        return c < rows_w ? wbase + (SWI && r >= kHalf ? N / 2 - kHalf : 0) + r : -1;
    };
    WmmaXTile<T, TR, G, false> xt;
    WT wt;
    auto k_loop = [&](auto tma_c) {
        constexpr int TMA = decltype(tma_c)::value;
        wmma_f32x8 acc[TMA][2] = {};
        xt.fetch(x, xoff, rows, 0);
        wt.fetch_mapped(q, scale, minv, row_of, K, 0);
        for (int64_t k0 = 0; k0 < K; k0 += kWmmaKC) {
            if (wt.win) {
                wt.stage_window(sw, mw);
                lds_barrier();
            }
            if constexpr (XH && COMBINE) xt.stage(xs, nullptr);
            else xt.stage_f16(xs);
            wt.stage_f16(wsm, sw, mw, k0);
            lds_barrier();
            if (k0 + kWmmaKC < K) {
                xt.fetch(x, xoff, rows, k0 + kWmmaKC);
                wt.fetch_mapped(q, scale, minv, row_of, K, k0 + kWmmaKC);
            }
#pragma unroll
            for (int kk = 0; kk < kWmmaKC; kk += 16) {
                wmma_f16x16 a[TMA], b[2];
#pragma unroll
                for (int i = 0; i < TMA; ++i) a[i] = wmma_load_rows_a16_f16(xs + 16 * (2 * i + wm) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
                for (int j = 0; j < 2; ++j) b[j] = wmma_load_rows_a16_f16(wsm + (wn * 32 + 16 * j) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) acc[i][j] = wmma_f16(a[i], b[j], acc[i][j]);
            }
            lds_barrier();
        }
        if (COMBINE && pbf16 && vec_out) {
#pragma unroll
            for (int i = 0; i < TMA; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int r = 16 * (2 * i + wm) + wmma_c_row(v), c = wn * 32 + 16 * j + (int)threadIdx.x % 16;
                        tiles_lds[r * kOutLd + c] = f32_to_bf16_bits(acc[i][j][v]);
                    }
            __syncthreads();
            uint16_t *pb = reinterpret_cast<uint16_t *>(partial);
            for (int e = (int)threadIdx.x; e < rows * (kWmmaBN / 8); e += kWmmaThreads) {
                const int r = e / (kWmmaBN / 8), c = e % (kWmmaBN / 8) * 8;
                if (c >= rows_w) continue;
                *reinterpret_cast<uint4 *>(pb + ooff[r] + n0 + c) =
                    *reinterpret_cast<const uint4 *>(tiles_lds + r * kOutLd + c);
            }
        }
    };
    switch ((nt + 1) / 2) {
        case 1: k_loop(std::integral_constant<int, 1>{}); break;
        case 2: k_loop(std::integral_constant<int, 2>{}); break;
        case 3: k_loop(std::integral_constant<int, 3>{}); break;
        default: k_loop(std::integral_constant<int, 4>{}); break;
    }
}

// 2. OPTIMIZED KERNEL: 2-Deep Software Pipelined (xa, xb and wa, wb)
template <typename T, int G, bool COMBINE, bool SWI, bool XH = false>
__global__ void __launch_bounds__(kWmmaThreads)
    grouped_wmma_f16_wide_pipe2_kernel(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                                       const int32_t *ws, int64_t A, int64_t N, int64_t K, T *y, float *partial, bool pbf16) {
    static_assert(!(SWI && COMBINE), "SwiGLU fuses into the gather only");
    using WT = WmmaWCodes<4, G, kWmmaBN>;
    constexpr int TR = RX, kHalf = kWmmaBN / 2, kTileElems = (TR + kWmmaBN) * kWmmaLd;
    constexpr int kOutLd = kWmmaBN + 8;
    __shared__ __attribute__((aligned(16))) uint16_t tiles_lds[kTileElems];
    uint16_t *xs = tiles_lds, *wsm = tiles_lds + TR * kWmmaLd;
    __shared__ uint16_t sw[kWmmaBN * WT::kWin], mw[kWmmaBN * WT::kWin];
    __shared__ int64_t xoff[TR], ooff[TR];
    const int32_t S = ws[1], E = ws[2];
    const int64_t Tmax = (int64_t)S / R + E + 2, WTmax = (int64_t)S / RW + E + 2, XTmax = (int64_t)S / RX + E + 2;
    const int32_t *offsets = ws + 4, *slots = offsets + E + 2;
    const int32_t *xtile_bucket = slots + S + 2 * Tmax + 2 * WTmax, *xtile_start = xtile_bucket + XTmax;
    const int64_t tile = blockIdx.y;
    if (tile >= xtile_start[XTmax]) return;
    const int32_t bucket = xtile_bucket[tile], st = xtile_start[tile];
    const int rows = min(TR, offsets[bucket + 1] - st), nt = (rows + 15) / 16;
    constexpr int kCols = SWI ? kHalf : kWmmaBN;
    const int64_t n0 = (int64_t)blockIdx.x * kCols, Nh = SWI ? N / 2 : N;
    const int rows_w = (int)(Nh - n0 < kCols ? Nh - n0 : kCols);
    if ((int)threadIdx.x < rows) {
        const int64_t slot = slots[st + threadIdx.x];
        xoff[threadIdx.x] = (COMBINE ? slot : slot / A) * K;
        ooff[threadIdx.x] = slot * (SWI ? N / 2 : N);
    }
    __syncthreads();
    if (bucket == E) {
        for (int i = threadIdx.x; i < rows * rows_w; i += kWmmaThreads) {
            const int r = i / rows_w, c = i % rows_w;
            if (COMBINE) put_partial(partial, ooff[r] + n0 + c, __int_as_float(0x7fc00000), pbf16);
        }
        return;
    }
    const int wave = (int)threadIdx.x / 32, wm = wave / 4, wn = wave % 4;
    const int64_t wbase = (int64_t)bucket * N + n0;
    const bool vec_out = N % 8 == 0 && reinterpret_cast<uintptr_t>(partial) % 16 == 0;
    const auto row_of = [&](int r) -> int64_t {
        const int c = SWI ? r % kHalf : r;
        return c < rows_w ? wbase + (SWI && r >= kHalf ? N / 2 - kHalf : 0) + r : -1;
    };
    WmmaXTile<T, TR, G, false> xa, xb;
    WT wa, wb;
    auto k_loop = [&](auto tma_c) {
        constexpr int TMA = decltype(tma_c)::value;
        wmma_f32x8 acc[TMA][2] = {};
        xa.fetch(x, xoff, rows, 0);
        wa.fetch_mapped(q, scale, minv, row_of, K, 0);
        if (kWmmaKC < K) {
            xb.fetch(x, xoff, rows, kWmmaKC);
            wb.fetch_mapped(q, scale, minv, row_of, K, kWmmaKC);
        }
        auto step = [&](WmmaXTile<T, TR, G, false> &cur_x, WT &cur_w, int64_t k0) {
            if (cur_w.win) {
                cur_w.stage_window(sw, mw);
                lds_barrier();
            }
            if constexpr (XH && COMBINE) cur_x.stage(xs, nullptr);
            else cur_x.stage_f16(xs);
            cur_w.stage_f16(wsm, sw, mw, k0);
            lds_barrier();
            if (k0 + 2 * kWmmaKC < K) {
                cur_x.fetch(x, xoff, rows, k0 + 2 * kWmmaKC);
                cur_w.fetch_mapped(q, scale, minv, row_of, K, k0 + 2 * kWmmaKC);
            }
#pragma unroll
            for (int kk = 0; kk < kWmmaKC; kk += 16) {
                wmma_f16x16 a[TMA], b[2];
#pragma unroll
                for (int i = 0; i < TMA; ++i) a[i] = wmma_load_rows_a16_f16(xs + 16 * (2 * i + wm) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
                for (int j = 0; j < 2; ++j) b[j] = wmma_load_rows_a16_f16(wsm + (wn * 32 + 16 * j) * kWmmaLd + kk, kWmmaLd);
#pragma unroll
                for (int i = 0; i < TMA; ++i)
#pragma unroll
                    for (int j = 0; j < 2; ++j) acc[i][j] = wmma_f16(a[i], b[j], acc[i][j]);
            }
            lds_barrier();
        };
        for (int64_t k0 = 0; k0 < K; k0 += 2 * kWmmaKC) {
            step(xa, wa, k0);
            if (k0 + kWmmaKC < K) step(xb, wb, k0 + kWmmaKC);
        }
        if (COMBINE && pbf16 && vec_out) {
#pragma unroll
            for (int i = 0; i < TMA; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
#pragma unroll
                    for (int v = 0; v < 8; ++v) {
                        const int r = 16 * (2 * i + wm) + wmma_c_row(v), c = wn * 32 + 16 * j + (int)threadIdx.x % 16;
                        tiles_lds[r * kOutLd + c] = f32_to_bf16_bits(acc[i][j][v]);
                    }
            __syncthreads();
            uint16_t *pb = reinterpret_cast<uint16_t *>(partial);
            for (int e = (int)threadIdx.x; e < rows * (kWmmaBN / 8); e += kWmmaThreads) {
                const int r = e / (kWmmaBN / 8), c = e % (kWmmaBN / 8) * 8;
                if (c >= rows_w) continue;
                *reinterpret_cast<uint4 *>(pb + ooff[r] + n0 + c) =
                    *reinterpret_cast<const uint4 *>(tiles_lds + r * kOutLd + c);
            }
        }
    };
    switch ((nt + 1) / 2) {
        case 1: k_loop(std::integral_constant<int, 1>{}); break;
        case 2: k_loop(std::integral_constant<int, 2>{}); break;
        case 3: k_loop(std::integral_constant<int, 3>{}); break;
        default: k_loop(std::integral_constant<int, 4>{}); break;
    }
}

}  // namespace

int main() {
    const int64_t T = 8192;
    const int64_t d = 2560;
    const int64_t inter = 640;
    const int64_t E = 512;
    const int64_t A = 10;
    const int64_t G = 128;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w((size_t)(E * d * inter));
    for (float &v : w) v = dist(rng);
    Q4Weight qw = quantize_q4(w.data(), E * d, inter, G);
    Q4Device down = Q4Device::upload(qw, "down");

    std::vector<uint16_t> host_h((size_t)(T * A * inter));
    for (size_t i = 0; i < host_h.size(); ++i) host_h[i] = 0x3c00;  // 1.0 in F16
    DeviceBuffer<uint16_t> h = DeviceBuffer<uint16_t>::from_host(host_h, "h");

    std::vector<int32_t> host_ids((size_t)(T * A));
    std::uniform_int_distribution<int32_t> exp_dist(0, (int32_t)(E - 1));
    for (int64_t i = 0; i < T * A; ++i) host_ids[i] = exp_dist(rng);
    DeviceBuffer<int32_t> route_ids = DeviceBuffer<int32_t>::from_host(host_ids, "route_ids");

    size_t ws_bytes = moe_route_workspace_bytes(T, A, E);
    DeviceBuffer<uint8_t> group_ws(ws_bytes, "group_ws");
    DeviceBuffer<uint32_t> err(3, "err");
    STRIX_HIP_CHECK(hipMemset(err.get(), 0, 12), "zero err");

    DeviceBuffer<float> partial_base((size_t)(T * A * d), "partial_base");
    DeviceBuffer<float> partial_pipe2((size_t)(T * A * d), "partial_pipe2");
    STRIX_HIP_CHECK(hipMemset(partial_base.get(), 0, T * A * d * 4), "zero");
    STRIX_HIP_CHECK(hipMemset(partial_pipe2.get(), 0, T * A * d * 4), "zero");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    moe_group_routes(route_ids.get(), T, A, E, group_ws.get(), ws_bytes, err.get(), stream);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync routes");

    const int64_t tiles = route_layout(T, A, E).XTmax;
    const int64_t cols = d, per_block = kWmmaBN;
    dim3 grid((unsigned)((cols + per_block - 1) / per_block), (unsigned)tiles), block(kWmmaThreads);

    // Warmup & Run both kernels
    grouped_wmma_f16_wide_baseline_kernel<uint16_t, 128, true, false, true><<<grid, block, 0, stream>>>(
        h.get(), down.view().q, down.view().scale, down.view().minv,
        reinterpret_cast<const int32_t *>(group_ws.get()), A, d, inter, nullptr, partial_base.get(), true);

    grouped_wmma_f16_wide_pipe2_kernel<uint16_t, 128, true, false, true><<<grid, block, 0, stream>>>(
        h.get(), down.view().q, down.view().scale, down.view().minv,
        reinterpret_cast<const int32_t *>(group_ws.get()), A, d, inter, nullptr, partial_pipe2.get(), true);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync runs");

    // Verification: compare BF16 partials on host
    std::printf("Checking 100%% Bit-Identity across all %ld BF16 elements...\n", (long)(T * A * d));
    std::vector<uint16_t> h_base((size_t)(T * A * d)), h_pipe2((size_t)(T * A * d));
    STRIX_HIP_CHECK(hipMemcpy(h_base.data(), partial_base.get(), h_base.size() * 2, hipMemcpyDeviceToHost), "cpy");
    STRIX_HIP_CHECK(hipMemcpy(h_pipe2.data(), partial_pipe2.get(), h_pipe2.size() * 2, hipMemcpyDeviceToHost), "cpy");

    int64_t diffs = 0;
    for (size_t i = 0; i < h_base.size(); ++i) {
        if (h_base[i] != h_pipe2[i]) ++diffs;
    }
    std::printf("Verification: %ld diffs / %zu total (%.6f%%)\n",
                (long)diffs, h_base.size(), (double)diffs / h_base.size() * 100.0);

    if (diffs == 0) {
        std::printf(">>> 100.000%% BIT-EXACT MATCH! <<<\n\n");
    } else {
        std::printf(">>> ERROR: BIT MISMATCH DETECTED! <<<\n\n");
    }

    // Benchmark over 30 iterations
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
        std::printf("Benchmark: %-30s -> %7.3f ms (48 layers = %6.2f ms)\n",
                    label, avg_ms, avg_ms * 48.0f);
        return avg_ms;
    };

    bench([&]() {
        grouped_wmma_f16_wide_baseline_kernel<uint16_t, 128, true, false, true><<<grid, block, 0, stream>>>(
            h.get(), down.view().q, down.view().scale, down.view().minv,
            reinterpret_cast<const int32_t *>(group_ws.get()), A, d, inter, nullptr, partial_base.get(), true);
    }, "Baseline Wide GEMM");

    bench([&]() {
        grouped_wmma_f16_wide_pipe2_kernel<uint16_t, 128, true, false, true><<<grid, block, 0, stream>>>(
            h.get(), down.view().q, down.view().scale, down.view().minv,
            reinterpret_cast<const int32_t *>(group_ws.get()), A, d, inter, nullptr, partial_pipe2.get(), true);
    }, "Pipe2 Wide GEMM");

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return 0;
}
