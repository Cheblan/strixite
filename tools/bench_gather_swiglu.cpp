#include "common/hip_runtime.hpp"
#include "kernels/linear.hpp"
#include "kernels/linear_q4.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/norm.hpp"
#include "kernels/qsplit.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"

#pragma clang diagnostic ignored "-Wunused-value"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <string>

using namespace strix;

namespace {

constexpr int kThreads = 256;
constexpr int kChunk = 32;

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

inline float host_bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    union { uint32_t u; float f; } c{u};
    return c.f;
}
__device__ inline float as_stored(float v, const uint16_t *) { return bf16_bits_to_f32(f32_to_bf16_bits(v)); }
__device__ inline float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }

__device__ inline void load_x32(const uint16_t *x, int64_t off, float (&v)[kChunk]) {
    const uint4 *p = reinterpret_cast<const uint4 *>(x + off);
#pragma unroll
    for (int i = 0; i < kChunk / 8; ++i) {
        uint4 u = p[i];
        uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * i + 2 * j] = __uint_as_float(w4[j] << 16);
            v[8 * i + 2 * j + 1] = __uint_as_float(w4[j] & 0xffff0000u);
        }
    }
}

__device__ inline float reduce_row(float v, float *part, int wave, int ks, int KS) {
    for (int off = warpSize / 2; off > 0; off >>= 1) v += __shfl_xor(v, off);
    if (KS == 1) return v;
    if (threadIdx.x % warpSize == 0) part[wave] = v;
    __syncthreads();
    if (ks != 0) return 0.f;
    float t = 0.f;
    for (int i = 0; i < KS; ++i) t += part[wave + i];
    return t;
}

__device__ inline bool expert_id_ok(int32_t id, int64_t E, uint32_t *err, int64_t slot) {
    if (id >= 0 && id < E) return true;
    if (err && atomicCAS(err, 0, 1) == 0) {
        err[1] = (uint32_t)id;
        err[2] = (uint32_t)slot;
    }
    return false;
}

// Packed BF16 dot2 support
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
__device__ inline float dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), c, false);
}
__device__ inline uint32_t code_pair(const uint32_t (&words)[4], int p) {
    const uint32_t b = (words[p / 4] >> (8 * (p % 4))) & 0xFFu;
    const float lo = (float)(b & 0xFu), hi = (float)(b >> 4);
    return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xffff0000u);
}
constexpr uint32_t kOnePair = 0x3F803F80u;

// -----------------------------------------------------------------------------
// Variant 2: Fused Gate+Up (scalar load of x, but single load of x & parallel weight loads)
// -----------------------------------------------------------------------------
template <typename T, int G, int HB>
__global__ void __launch_bounds__(kThreads)
kernel_fused_gateup(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                    const int32_t *ids, int64_t E, int64_t A, T *y, int64_t N, int64_t K,
                    int KS, uint32_t *err) {
    __shared__ float part_g[kThreads / 32], part_u[kThreads / 32];
    const int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize, ks = wave % KS;
    const int64_t I = N / 2, j = (int64_t)blockIdx.x * (blockDim.x / warpSize / KS) + wave / KS;
    const int64_t slot = (int64_t)blockIdx.z * A + blockIdx.y;
    const int32_t id = ids[slot];
    const bool ok = id >= 0 && id < E;
    float vg = 0.f, vu = 0.f;

    if (ok && j < I) {
        const T *xr = x + (int64_t)blockIdx.z * K;
        const int64_t rg = (int64_t)id * N + j, ru = rg + I;
        const uint8_t *q_g = q + rg * kernels::qsplit_row_bytes<HB>(K);
        const uint8_t *q_u = q + ru * kernels::qsplit_row_bytes<HB>(K);
        const uint16_t *s_g_ptr = scale + rg * (K / G);
        const uint16_t *s_u_ptr = scale + ru * (K / G);
        const uint16_t *m_g_ptr = minv + rg * (K / G);
        const uint16_t *m_u_ptr = minv + ru * (K / G);
        const int64_t chunks = K / kChunk;

        for (int64_t c = (int64_t)ks * warpSize + lane; c < chunks; c += (int64_t)KS * warpSize) {
            uint4 packed_g = reinterpret_cast<const uint4 *>(q_g)[c];
            uint4 packed_u = reinterpret_cast<const uint4 *>(q_u)[c];
            const float s_g = bf16_bits_to_f32(s_g_ptr[c * kChunk / G]);
            const float s_u = bf16_bits_to_f32(s_u_ptr[c * kChunk / G]);
            const float mn_g = bf16_bits_to_f32(m_g_ptr[c * kChunk / G]);
            const float mn_u = bf16_bits_to_f32(m_u_ptr[c * kChunk / G]);

            float xv[kChunk];
            load_x32(xr, c * kChunk, xv);

            uint32_t words_g[4] = {packed_g.x, packed_g.y, packed_g.z, packed_g.w};
            uint32_t words_u[4] = {packed_u.x, packed_u.y, packed_u.z, packed_u.w};
            const kernels::QHigh32<HB> hb_g = kernels::QHigh32<HB>::load(q_g, K, c * kChunk);
            const kernels::QHigh32<HB> hb_u = kernels::QHigh32<HB>::load(q_u, K, c * kChunk);

            float dot_g = 0.f, dot_u = 0.f, xsum = 0.f;
#pragma unroll
            for (int i = 0; i < kChunk; ++i) {
                float xi = xv[i];
                dot_g = fmaf(xi, (float)(((words_g[i / 8] >> (4 * (i % 8))) & 0xFu) | hb_g.at(i)), dot_g);
                dot_u = fmaf(xi, (float)(((words_u[i / 8] >> (4 * (i % 8))) & 0xFu) | hb_u.at(i)), dot_u);
                xsum += xi;
            }
            vg = fmaf(s_g, dot_g, fmaf(mn_g, xsum, vg));
            vu = fmaf(s_u, dot_u, fmaf(mn_u, xsum, vu));
        }
    }
    vg = reduce_row(vg, part_g, wave, ks, KS);
    vu = reduce_row(vu, part_u, wave, ks, KS);
    if (lane == 0 && ks == 0 && j < I) {
        if (!ok && j == 0) expert_id_ok(id, E, err, slot);
        const float g = as_stored(vg, y), u = as_stored(vu, y);
        const float hv = as_stored(silu(g), y) * u;
        store(y, slot * I + j, ok ? hv : __int_as_float(0x7fc00000));
    }
}

// -----------------------------------------------------------------------------
// Variant 4: Packed dot2 Fused Gate+Up (uses v_dot2_f32_bf16)
// -----------------------------------------------------------------------------
template <typename T, int G, int HB>
__global__ void __launch_bounds__(kThreads)
kernel_dot2_fused_gateup(const T *x, const uint8_t *q, const uint16_t *scale, const uint16_t *minv,
                         const int32_t *ids, int64_t E, int64_t A, T *y, int64_t N, int64_t K,
                         int KS, uint32_t *err) {
    __shared__ float part_g[kThreads / 32], part_u[kThreads / 32];
    const int lane = threadIdx.x % warpSize, wave = threadIdx.x / warpSize, ks = wave % KS;
    const int64_t I = N / 2, j = (int64_t)blockIdx.x * (blockDim.x / warpSize / KS) + wave / KS;
    const int64_t slot = (int64_t)blockIdx.z * A + blockIdx.y;
    const int32_t id = ids[slot];
    const bool ok = id >= 0 && id < E;
    float vg = 0.f, vu = 0.f;

    if (ok && j < I) {
        const T *xr = x + (int64_t)blockIdx.z * K;
        const int64_t rg = (int64_t)id * N + j, ru = rg + I;
        const uint8_t *q_g = q + rg * kernels::qsplit_row_bytes<HB>(K);
        const uint8_t *q_u = q + ru * kernels::qsplit_row_bytes<HB>(K);
        const uint16_t *s_g_ptr = scale + rg * (K / G);
        const uint16_t *s_u_ptr = scale + ru * (K / G);
        const uint16_t *m_g_ptr = minv + rg * (K / G);
        const uint16_t *m_u_ptr = minv + ru * (K / G);
        const int64_t chunks = K / kChunk;

        for (int64_t c = (int64_t)ks * warpSize + lane; c < chunks; c += (int64_t)KS * warpSize) {
            uint4 packed_g = reinterpret_cast<const uint4 *>(q_g)[c];
            uint4 packed_u = reinterpret_cast<const uint4 *>(q_u)[c];
            const float s_g = bf16_bits_to_f32(s_g_ptr[c * kChunk / G]);
            const float s_u = bf16_bits_to_f32(s_u_ptr[c * kChunk / G]);
            const float mn_g = bf16_bits_to_f32(m_g_ptr[c * kChunk / G]);
            const float mn_u = bf16_bits_to_f32(m_u_ptr[c * kChunk / G]);

            uint32_t words_g[4] = {packed_g.x, packed_g.y, packed_g.z, packed_g.w};
            uint32_t words_u[4] = {packed_u.x, packed_u.y, packed_u.z, packed_u.w};

            uint32_t cp_g[kChunk / 2], cp_u[kChunk / 2];
#pragma unroll
            for (int p = 0; p < kChunk / 2; ++p) {
                cp_g[p] = code_pair(words_g, p);
                cp_u[p] = code_pair(words_u, p);
            }

            const uint4 *xp = reinterpret_cast<const uint4 *>(xr + c * kChunk);
            float dot_g = 0.f, dot_u = 0.f, xsum = 0.f;
#pragma unroll
            for (int i = 0; i < kChunk / 8; ++i) {
                const uint4 u4 = xp[i];
                const uint32_t xw[4] = {u4.x, u4.y, u4.z, u4.w};
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    dot_g = dot2(xw[p], cp_g[4 * i + p], dot_g);
                    dot_u = dot2(xw[p], cp_u[4 * i + p], dot_u);
                    xsum = dot2(xw[p], kOnePair, xsum);
                }
            }
            vg = fmaf(s_g, dot_g, fmaf(mn_g, xsum, vg));
            vu = fmaf(s_u, dot_u, fmaf(mn_u, xsum, vu));
        }
    }
    vg = reduce_row(vg, part_g, wave, ks, KS);
    vu = reduce_row(vu, part_u, wave, ks, KS);
    if (lane == 0 && ks == 0 && j < I) {
        if (!ok && j == 0) expert_id_ok(id, E, err, slot);
        const float g = as_stored(vg, y), u = as_stored(vu, y);
        const float hv = as_stored(silu(g), y) * u;
        store(y, slot * I + j, ok ? hv : __int_as_float(0x7fc00000));
    }
}

}  // namespace

int main(int argc, char **argv) {
    hipSetDevice(0);
    hipStream_t stream;
    hipStreamCreate(&stream);

    // Qwen 3.8-Flash-Next Architecture dimensions
    const int64_t K = 2560;
    const int64_t E = 512;
    const int64_t I = 640;
    const int64_t N = 2 * I;  // 1280 (gate + up)
    const int64_t G = 128;
    const int64_t A = 10;
    const int64_t max_M = 6;

    printf("================================================================================\n");
    printf(" AMD Strix Halo gfx1151 MoE Gather SwiGLU Microbenchmark\n");
    printf(" Model: Qwen 3.8-Flash-Next | E=%ld, A=%ld, K=%ld, I=%ld, N=%ld, G=%ld\n", E, A, K, I, N, G);
    printf("================================================================================\n\n");

    // Allocate Device Buffers
    const int64_t total_rows = E * N;
    const int64_t q_bytes = total_rows * (K / 2);
    const int64_t s_bytes = total_rows * (K / G) * sizeof(uint16_t);
    const int64_t m_bytes = total_rows * (K / G) * sizeof(uint16_t);

    printf("Allocating Layer Expert Weights:\n");
    printf("  q:     %.2f MB\n", q_bytes / 1e6);
    printf("  scale: %.2f MB\n", s_bytes / 1e6);
    printf("  minv:  %.2f MB\n", m_bytes / 1e6);
    printf("  Total: %.2f MB\n\n", (q_bytes + s_bytes + m_bytes) / 1e6);

    uint8_t *d_q = nullptr;
    uint16_t *d_s = nullptr, *d_m = nullptr;
    hipMalloc(&d_q, q_bytes);
    hipMalloc(&d_s, s_bytes);
    hipMalloc(&d_m, m_bytes);

    // Initialize with dummy values
    hipMemset(d_q, 0x55, q_bytes);
    hipMemset(d_s, 0x3c, s_bytes);
    hipMemset(d_m, 0x38, m_bytes);

    Q4DeviceView w_view{d_q, d_s, d_m, total_rows, K, G};

    // Activations x, Output y, and IDs
    uint16_t *d_x = nullptr, *d_y_base = nullptr, *d_y_test = nullptr;
    int32_t *d_ids = nullptr;
    uint32_t *d_err = nullptr;

    hipMalloc(&d_x, max_M * K * sizeof(uint16_t));
    hipMalloc(&d_y_base, max_M * A * I * sizeof(uint16_t));
    hipMalloc(&d_y_test, max_M * A * I * sizeof(uint16_t));
    hipMalloc(&d_ids, max_M * A * sizeof(int32_t));
    hipMalloc(&d_err, 3 * sizeof(uint32_t));

    hipMemset(d_x, 0x3c, max_M * K * sizeof(uint16_t));
    hipMemset(d_err, 0, 3 * sizeof(uint32_t));

    // Fill IDs with distinct expert IDs: e.g. 10 * m + a
    std::vector<int32_t> h_ids(max_M * A);
    for (int m = 0; m < max_M; ++m) {
        for (int a = 0; a < A; ++a) {
            h_ids[m * A + a] = (m * A + a) % E;
        }
    }
    hipMemcpy(d_ids, h_ids.data(), max_M * A * sizeof(int32_t), hipMemcpyHostToDevice);

    // Host buffers for bit-exactness check
    std::vector<uint16_t> h_y_base(max_M * A * I);
    std::vector<uint16_t> h_y_test(max_M * A * I);

    // Benchmarking helper
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // To measure pure DRAM bandwidth without MALL/L3 cache hits (32 MB cache),
    // we allocate an array of expert IDs for 48 distinct iterations (one per layer),
    // each reading distinct experts from the 891 MB weights table in DRAM!
    const int kDramBatches = 48;
    std::vector<int32_t> h_ids_dram(kDramBatches * max_M * A);
    for (int b = 0; b < kDramBatches; ++b) {
        for (int m = 0; m < max_M; ++m) {
            for (int a = 0; a < A; ++a) {
                // Pick unique experts across batches
                h_ids_dram[b * (max_M * A) + m * A + a] = ((b * max_M + m) * A + a) % E;
            }
        }
    }
    int32_t *d_ids_dram = nullptr;
    hipMalloc(&d_ids_dram, kDramBatches * max_M * A * sizeof(int32_t));
    hipMemcpy(d_ids_dram, h_ids_dram.data(), kDramBatches * max_M * A * sizeof(int32_t), hipMemcpyHostToDevice);

    auto bench_dram = [&](const char *name, auto launch_with_ids, int64_t M) {
        // Warmup
        for (int b = 0; b < 4; ++b) {
            launch_with_ids(d_ids_dram + b * (max_M * A), stream);
        }
        hipStreamSynchronize(stream);

        // Timing 48 distinct batches (simulating all 48 layers reading distinct DRAM weights)
        hipEventRecord(start, stream);
        for (int b = 0; b < kDramBatches; ++b) {
            launch_with_ids(d_ids_dram + b * (max_M * A), stream);
        }
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);

        float ms = 0.f;
        hipEventElapsedTime(&ms, start, stop);
        float total_48l_ms = ms;
        float avg_us_per_layer = (ms * 1000.f) / kDramBatches;
        
        const double bytes_read = (double)M * A * 1740800.0;
        const double bw_gbs = (bytes_read / 1e9) / (avg_us_per_layer / 1e6);

        printf("  %-32s | %6.2f us | 48L: %5.2f ms | %6.1f GB/s\n",
               name, avg_us_per_layer, total_48l_ms, bw_gbs);
        return bw_gbs;
    };

    // Allocate Down Projection Expert Weights for Combine benchmark
    // E = 512, N_down = 2560, K_down = 640, G = 128
    const int64_t N_down = 2560;
    const int64_t K_down = 640;
    const int64_t total_rows_down = E * N_down;
    const int64_t q_down_bytes = total_rows_down * (K_down / 2);
    const int64_t s_down_bytes = total_rows_down * (K_down / G) * sizeof(uint16_t);
    const int64_t m_down_bytes = total_rows_down * (K_down / G) * sizeof(uint16_t);

    uint8_t *d_q_down = nullptr;
    uint16_t *d_s_down = nullptr, *d_m_down = nullptr;
    hipMalloc(&d_q_down, q_down_bytes);
    hipMalloc(&d_s_down, s_down_bytes);
    hipMalloc(&d_m_down, m_down_bytes);
    hipMemset(d_q_down, 0x55, q_down_bytes);
    hipMemset(d_s_down, 0x3c, s_down_bytes);
    hipMemset(d_m_down, 0x38, m_down_bytes);
    Q4DeviceView w_down_view{d_q_down, d_s_down, d_m_down, total_rows_down, K_down, G};

    float *d_coef = nullptr;
    hipMalloc(&d_coef, max_M * A * sizeof(float));
    std::vector<float> h_coef(max_M * A, 0.1f);
    hipMemcpy(d_coef, h_coef.data(), max_M * A * sizeof(float), hipMemcpyHostToDevice);

    uint16_t *d_y_comb = nullptr;
    hipMalloc(&d_y_comb, max_M * N_down * sizeof(uint16_t));

    auto bench_combine_dram = [&](const char *name, int64_t M) {
        // Timing 48 distinct batches
        hipEventRecord(start, stream);
        for (int b = 0; b < kDramBatches; ++b) {
            kernels::linear_q4_experts_combine(d_y_test, w_down_view, E, d_ids_dram + b * (max_M * A),
                                               d_coef, M, A, d_y_comb, kernels::Act::BF16, d_err, stream);
        }
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);

        float ms = 0.f;
        hipEventElapsedTime(&ms, start, stop);
        float avg_us = (ms * 1000.f) / kDramBatches;
        const double bytes_read = (double)M * A * (N_down * (K_down / 2 + (K_down / G) * 4));
        const double bw_gbs = (bytes_read / 1e9) / (avg_us / 1e6);

        printf("  %-32s | %6.2f us | 48L: %5.2f ms | %6.1f GB/s\n",
               name, avg_us, ms, bw_gbs);
    };

    std::vector<int64_t> m_values = {1, 2, 3, 4, 6};
    for (int64_t M : m_values) {
        printf("\n==================== TEST BATCH M = %ld (Active Slots = %ld, Weights Read = %.2f MB) ====================\n",
               M, M * A, (M * A * 1740800.0) / 1e6);

        // 1. Baseline (KS = 1)
        auto launch_baseline = [&](const int32_t *cur_ids, hipStream_t s) {
            kernels::linear_q4_experts_gather_swiglu(d_x, w_view, E, cur_ids, M, A, d_y_base, kernels::Act::BF16, d_err, s);
        };
        launch_baseline(d_ids, stream);
        hipStreamSynchronize(stream);
        hipMemcpy(h_y_base.data(), d_y_base, M * A * I * sizeof(uint16_t), hipMemcpyDeviceToHost);

        bench_dram("Baseline (Current KS=1)", launch_baseline, M);

        // 2. Fused Gate+Up (KS=1)
        auto launch_fused_ks1 = [&](const int32_t *cur_ids, hipStream_t s) {
            const int KS = 1;
            const int waves = kThreads / 32;
            const int64_t rows_per_block = waves / KS;
            const unsigned bx = (unsigned)((I + rows_per_block - 1) / rows_per_block);
            dim3 grid(bx, (unsigned)A, (unsigned)M);
            kernel_fused_gateup<uint16_t, 128, 0><<<grid, kThreads, 0, s>>>(
                d_x, d_q, d_s, d_m, cur_ids, E, A, d_y_test, N, K, KS, d_err);
        };
        launch_fused_ks1(d_ids, stream);
        hipStreamSynchronize(stream);
        hipMemcpy(h_y_test.data(), d_y_test, M * A * I * sizeof(uint16_t), hipMemcpyDeviceToHost);
        {
            int mismatches = 0;
            for (int64_t i = 0; i < M * A * I; ++i) {
                if (h_y_test[i] != h_y_base[i]) mismatches++;
            }
            if (mismatches == 0) printf("    [Verify Fused KS=1] 100.000%% BIT-EXACT MATCH!\n");
            else printf("    [Verify Fused KS=1] MISMATCH: %d values\n", mismatches);
        }
        bench_dram("Fused Gate+Up (KS=1)", launch_fused_ks1, M);

        // 3. Fused Gate+Up (KS=2)
        auto launch_fused_ks2 = [&](const int32_t *cur_ids, hipStream_t s) {
            const int KS = 2;
            const int waves = kThreads / 32;
            const int64_t rows_per_block = waves / KS;
            const unsigned bx = (unsigned)((I + rows_per_block - 1) / rows_per_block);
            dim3 grid(bx, (unsigned)A, (unsigned)M);
            kernel_fused_gateup<uint16_t, 128, 0><<<grid, kThreads, 0, s>>>(
                d_x, d_q, d_s, d_m, cur_ids, E, A, d_y_test, N, K, KS, d_err);
        };
        bench_dram("Fused Gate+Up (KS=2)", launch_fused_ks2, M);

        // 4. Packed dot2 Fused Gate+Up (KS=1)
        auto launch_dot2_ks1 = [&](const int32_t *cur_ids, hipStream_t s) {
            const int KS = 1;
            const int waves = kThreads / 32;
            const int64_t rows_per_block = waves / KS;
            const unsigned bx = (unsigned)((I + rows_per_block - 1) / rows_per_block);
            dim3 grid(bx, (unsigned)A, (unsigned)M);
            kernel_dot2_fused_gateup<uint16_t, 128, 0><<<grid, kThreads, 0, s>>>(
                d_x, d_q, d_s, d_m, cur_ids, E, A, d_y_test, N, K, KS, d_err);
        };
        launch_dot2_ks1(d_ids, stream);
        hipStreamSynchronize(stream);
        hipMemcpy(h_y_test.data(), d_y_test, M * A * I * sizeof(uint16_t), hipMemcpyDeviceToHost);
        {
            int mismatches = 0;
            for (int64_t i = 0; i < M * A * I; ++i) {
                if (h_y_test[i] != h_y_base[i]) mismatches++;
            }
            if (mismatches == 0) printf("    [Verify Packed dot2 KS=1] 100.000%% BIT-EXACT MATCH!\n");
            else printf("    [Verify Packed dot2 KS=1] MISMATCH: %d values\n", mismatches);
        }
        bench_dram("Packed dot2 Fused (KS=1)", launch_dot2_ks1, M);

        // 5. Packed dot2 Fused Gate+Up (KS=2)
        auto launch_dot2_ks2 = [&](const int32_t *cur_ids, hipStream_t s) {
            const int KS = 2;
            const int waves = kThreads / 32;
            const int64_t rows_per_block = waves / KS;
            const unsigned bx = (unsigned)((I + rows_per_block - 1) / rows_per_block);
            dim3 grid(bx, (unsigned)A, (unsigned)M);
            kernel_dot2_fused_gateup<uint16_t, 128, 0><<<grid, kThreads, 0, s>>>(
                d_x, d_q, d_s, d_m, cur_ids, E, A, d_y_test, N, K, KS, d_err);
        };
        bench_dram("Packed dot2 Fused (KS=2)", launch_dot2_ks2, M);

        // 6. Combine Baseline
        bench_combine_dram("Combine Down (Baseline)", M);

        // 7. Full MoE Components Breakdown
        // Measure Router (linear_bf16w: 513 x 2560)
        {
            const int64_t router_N = 513;
            const int64_t router_K = 2560;
            hipEventRecord(start, stream);
            for (int b = 0; b < kDramBatches; ++b) {
                kernels::linear_bf16w(d_x, (const uint16_t *)d_q, d_y_comb, M, router_N, router_K,
                                      kernels::Act::BF16, kernels::Act::F32, stream);
            }
            hipEventRecord(stop, stream);
            hipEventSynchronize(stop);
            float ms = 0.f;
            hipEventElapsedTime(&ms, start, stop);
            float avg_us = (ms * 1000.f) / kDramBatches;
            double bw = (router_N * router_K * 2.0 / 1e9) / (avg_us / 1e6);
            printf("  %-32s | %6.2f us | 48L: %5.2f ms | %6.1f GB/s\n",
                   "Router GEMV (513x2560)", avg_us, ms, bw);
        }

        // Measure Shared Expert (gate_up Q4 + swiglu + down Q4)
        {
            Q4DeviceView w_sh_gate{d_q, d_s, d_m, 1280, K, G};
            Q4DeviceView w_sh_down{d_q_down, d_s_down, d_m_down, 2560, 640, G};
            hipEventRecord(start, stream);
            for (int b = 0; b < kDramBatches; ++b) {
                kernels::linear_q4(d_x, w_sh_gate, d_y_comb, M, kernels::Act::BF16, stream);
                kernels::swiglu(d_y_comb, d_x, M, 640, kernels::Act::BF16, stream);
                kernels::linear_q4(d_x, w_sh_down, d_y_comb, M, kernels::Act::BF16, stream);
            }
            hipEventRecord(stop, stream);
            hipEventSynchronize(stop);
            float ms = 0.f;
            hipEventElapsedTime(&ms, start, stop);
            float avg_us = (ms * 1000.f) / kDramBatches;
            double bw = ((1740800.0 + 870400.0) / 1e9) / (avg_us / 1e6);
            printf("  %-32s | %6.2f us | 48L: %5.2f ms | %6.1f GB/s\n",
                   "Shared Expert (Gate/Up+Swi+Down)", avg_us, ms, bw);
        }
    }

    printf("\nBenchmark completed successfully.\n");

    hipFree(d_q);
    hipFree(d_s);
    hipFree(d_m);
    hipFree(d_x);
    hipFree(d_y_base);
    hipFree(d_y_test);
    hipFree(d_ids);
    hipFree(d_err);
    hipStreamDestroy(stream);

    return 0;
}
