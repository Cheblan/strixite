#include "kernels/linear_q5.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"

#include <cstdint>
#include <type_traits>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;
constexpr int kRowsPerPass = 8;
constexpr int kWideRowsPerPass = 16;
constexpr int kChunk = 32;       // 32 codes per load: 16 bytes low nibbles + 4 bytes high bits = 20 bytes
constexpr int64_t kMaxN = 0x7fffffff;
constexpr int64_t kMaxM = 65535ll * kRowsPerPass;
constexpr int64_t kMaxK = 1 << 24;

__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }

__device__ inline void load_x32(const float *x, int64_t off, float (&v)[kChunk]) {
    const float4 *p = reinterpret_cast<const float4 *>(x + off);
#pragma unroll
    for (int i = 0; i < kChunk / 4; ++i) {
        const float4 f = p[i];
        v[4 * i] = f.x, v[4 * i + 1] = f.y, v[4 * i + 2] = f.z, v[4 * i + 3] = f.w;
    }
}
__device__ inline void load_x32(const uint16_t *x, int64_t off, float (&v)[kChunk]) {
    const uint4 *p = reinterpret_cast<const uint4 *>(x + off);
#pragma unroll
    for (int i = 0; i < kChunk / 8; ++i) {
        const uint4 u = p[i];
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * i + 2 * j] = __uint_as_float(w4[j] << 16);
            v[8 * i + 2 * j + 1] = __uint_as_float(w4[j] & 0xffff0000u);
        }
    }
}

struct HighBitsQ5 {
    uint32_t w;
    __device__ static HighBitsQ5 load(const uint8_t *plane, int64_t k) {
        HighBitsQ5 h;
        h.w = *reinterpret_cast<const uint32_t *>(plane + k / 8);
        return h;
    }
    __device__ uint32_t at(int i) const {
        return (w >> i) & 0x1u;
    }
};

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
__device__ inline float dot2(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, a), __builtin_bit_cast(bf16x2, b), c, false);
}
constexpr uint32_t kOnePair = 0x3F803F80u;

__device__ inline uint32_t code_pair_q5(const uint32_t (&words)[4], const HighBitsQ5 &hb, int p) {
    const uint32_t b = (words[p / 4] >> (8 * (p % 4))) & 0xFFu;
    const uint32_t lo_nibble = b & 0xFu;
    const uint32_t hi_nibble = b >> 4;
    const uint32_t hn = (hb.w >> (2 * p)) & 0x3u;
    const uint32_t h0 = hn & 0x1u;
    const uint32_t h1 = hn >> 1;
    const float c0 = (float)(lo_nibble | (h0 << 4));
    const float c1 = (float)(hi_nibble | (h1 << 4));
    return (__float_as_uint(c0) >> 16) | (__float_as_uint(c1) & 0xffff0000u);
}

template <typename TX, typename TY, int G, int MR>
__global__ void __launch_bounds__(kThreads) linear_q5_kernel(const TX *x, const uint8_t *q, const uint16_t *scale,
                                                            const uint16_t *minv, TY *y, int64_t M, int64_t N,
                                                            int64_t K, int KS) {
    __shared__ float part[kThreads / 32][MR];
    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    const int64_t n = (int64_t)blockIdx.x * (blockDim.x / warpSize / KS) + wave / KS;
    const int ks = wave % KS;
    const int64_t m0 = (int64_t)blockIdx.y * MR;
    const int rows = (int)((M - m0) < MR ? (M - m0) : MR);

    float acc[MR];
#pragma unroll
    for (int r = 0; r < MR; ++r) acc[r] = 0.f;

    if (n < N) {
        const uint8_t *row = q + n * (K / 2 + K * 1 / 8);  // K * 5/8 row bytes
        const uint4 *lo_row = reinterpret_cast<const uint4 *>(row);
        const uint8_t *hi_plane = row + K / 2;
        const uint16_t *srow = scale + n * (K / G);
        const uint16_t *mrow = minv + n * (K / G);
        for (int64_t k = ((int64_t)ks * warpSize + lane) * kChunk; k < K; k += (int64_t)KS * warpSize * kChunk) {
            if constexpr (std::is_same_v<TX, uint16_t>) {
                if constexpr (MR <= 8) {
                    uint4 u0[MR], u1[MR], u2[MR], u3[MR];
#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        if (r < rows) {
                            const uint4 *xp = reinterpret_cast<const uint4 *>(x + (m0 + r) * K + k);
                            u0[r] = xp[0];
                            u1[r] = xp[1];
                            u2[r] = xp[2];
                            u3[r] = xp[3];
                        }
                    }

                    const uint4 lo = lo_row[k / kChunk];
                    const HighBitsQ5 hb = HighBitsQ5::load(hi_plane, k);
                    const uint32_t words[4] = {lo.x, lo.y, lo.z, lo.w};
                    const float s = bf16_bits_to_f32(srow[k / G]);
                    const float mn = bf16_bits_to_f32(mrow[k / G]);

                    uint32_t cp[kChunk / 2];
#pragma unroll
                    for (int p = 0; p < kChunk / 2; ++p) cp[p] = code_pair_q5(words, hb, p);

#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        if (r < rows) {
                            const uint32_t xw0[4] = {u0[r].x, u0[r].y, u0[r].z, u0[r].w};
                            const uint32_t xw1[4] = {u1[r].x, u1[r].y, u1[r].z, u1[r].w};
                            const uint32_t xw2[4] = {u2[r].x, u2[r].y, u2[r].z, u2[r].w};
                            const uint32_t xw3[4] = {u3[r].x, u3[r].y, u3[r].z, u3[r].w};
                            float dot = 0.f, xsum = 0.f;
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dot = dot2(xw0[j], cp[j], dot);
                                xsum = dot2(xw0[j], kOnePair, xsum);
                            }
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dot = dot2(xw1[j], cp[4 + j], dot);
                                xsum = dot2(xw1[j], kOnePair, xsum);
                            }
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dot = dot2(xw2[j], cp[8 + j], dot);
                                xsum = dot2(xw2[j], kOnePair, xsum);
                            }
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dot = dot2(xw3[j], cp[12 + j], dot);
                                xsum = dot2(xw3[j], kOnePair, xsum);
                            }
                            acc[r] = fmaf(s, dot, fmaf(mn, xsum, acc[r]));
                        }
                    }
                } else {
                    const uint4 lo = lo_row[k / kChunk];
                    const HighBitsQ5 hb = HighBitsQ5::load(hi_plane, k);
                    const uint32_t words[4] = {lo.x, lo.y, lo.z, lo.w};
                    const float s = bf16_bits_to_f32(srow[k / G]);
                    const float mn = bf16_bits_to_f32(mrow[k / G]);

                    uint32_t cp[kChunk / 2];
#pragma unroll
                    for (int p = 0; p < kChunk / 2; ++p) cp[p] = code_pair_q5(words, hb, p);

#pragma unroll
                    for (int r = 0; r < MR; ++r) {
                        if (r < rows) {
                            const uint4 *xp = reinterpret_cast<const uint4 *>(x + (m0 + r) * K + k);
                            const uint4 u0_val = xp[0];
                            const uint4 u1_val = xp[1];
                            const uint4 u2_val = xp[2];
                            const uint4 u3_val = xp[3];
                            const uint32_t xw0[4] = {u0_val.x, u0_val.y, u0_val.z, u0_val.w};
                            const uint32_t xw1[4] = {u1_val.x, u1_val.y, u1_val.z, u1_val.w};
                            const uint32_t xw2[4] = {u2_val.x, u2_val.y, u2_val.z, u2_val.w};
                            const uint32_t xw3[4] = {u3_val.x, u3_val.y, u3_val.z, u3_val.w};
                            float dot = 0.f, xsum = 0.f;
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dot = dot2(xw0[j], cp[j], dot);
                                xsum = dot2(xw0[j], kOnePair, xsum);
                            }
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dot = dot2(xw1[j], cp[4 + j], dot);
                                xsum = dot2(xw1[j], kOnePair, xsum);
                            }
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dot = dot2(xw2[j], cp[8 + j], dot);
                                xsum = dot2(xw2[j], kOnePair, xsum);
                            }
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                dot = dot2(xw3[j], cp[12 + j], dot);
                                xsum = dot2(xw3[j], kOnePair, xsum);
                            }
                            acc[r] = fmaf(s, dot, fmaf(mn, xsum, acc[r]));
                        }
                    }
                }
                continue;
            }

            // Fallback for non-BF16 activations (float)
            float xv[MR][kChunk];
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                if (r < rows) load_x32(x, (m0 + r) * K + k, xv[r]);
            }
            const uint4 lo = lo_row[k / kChunk];
            const HighBitsQ5 hb = HighBitsQ5::load(hi_plane, k);
            const uint32_t lw[4] = {lo.x, lo.y, lo.z, lo.w};
            const float s = bf16_bits_to_f32(srow[k / G]);
            const float mn = bf16_bits_to_f32(mrow[k / G]);
#pragma unroll
            for (int r = 0; r < MR; ++r) {
                if (r < rows) {
                    float dot = 0.f, xsum = 0.f;
#pragma unroll
                    for (int i = 0; i < kChunk; ++i) {
                        const uint32_t c = ((lw[i / 8] >> (4 * (i % 8))) & 0xFu) | (hb.at(i) << 4);
                        dot = fmaf(xv[r][i], (float)c, dot);
                        xsum += xv[r][i];
                    }
                    acc[r] = fmaf(s, dot, fmaf(mn, xsum, acc[r]));
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MR; ++r)
        for (int off = warpSize / 2; off > 0; off >>= 1) acc[r] += __shfl_xor(acc[r], off);
    if (KS == 1) {
        if (lane == 0 && n < N)
            for (int r = 0; r < rows; ++r) store(y, (m0 + r) * N + n, acc[r]);
        return;
    }
    if (lane == 0)
#pragma unroll
        for (int r = 0; r < MR; ++r) part[wave][r] = acc[r];
    __syncthreads();
    if (lane == 0 && ks == 0 && n < N)
        for (int r = 0; r < rows; ++r) {
            float v = 0.f;
            for (int i = 0; i < KS; ++i) v += part[wave + i][r];
            store(y, (m0 + r) * N + n, v);
        }
}

int split_k(int64_t K) {
    int ks = 1;
    while (ks < 8 && K / (2048 * (ks * 2)) >= 1) ks *= 2;
    return ks;
}

template <typename TX, typename TY, int MR>
void launch_mr(const char *fn, const TX *x, const Q5DeviceView &w, TY *y, int64_t M, int KS, hipStream_t stream) {
    const int waves = kThreads / device_warp_size(kThreads);
    STRIX_CHECK(KS >= 1 && waves % KS == 0, fn, ": split-K factor ", KS, " must divide the ", waves, " waves of a block");
    const int64_t rows_per_block = waves / KS;
    const dim3 grid((unsigned)((w.N + rows_per_block - 1) / rows_per_block), (unsigned)((M + MR - 1) / MR)),
        block(kThreads);
    switch (w.G) {
        case 32: linear_q5_kernel<TX, TY, 32, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        case 64: linear_q5_kernel<TX, TY, 64, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        case 128: linear_q5_kernel<TX, TY, 128, MR><<<grid, block, 0, stream>>>(x, w.q, w.scale, w.minv, y, M, w.N, w.K, KS); break;
        default: STRIX_FAIL(fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    }
}

template <typename TX, typename TY>
void launch(const char *fn, const TX *x, const Q5DeviceView &w, TY *y, int64_t M, hipStream_t stream) {
    if (M == 1) launch_mr<TX, TY, 1>(fn, x, w, y, M, split_k(w.K), stream);
    else if (M == 2) launch_mr<TX, TY, 2>(fn, x, w, y, M, split_k(w.K), stream);
    else if (M == 3) launch_mr<TX, TY, 3>(fn, x, w, y, M, split_k(w.K), stream);
    else if (M == 4) launch_mr<TX, TY, 4>(fn, x, w, y, M, split_k(w.K), stream);
    else if (M == 5) launch_mr<TX, TY, 5>(fn, x, w, y, M, split_k(w.K), stream);
    else if (M <= kRowsPerPass) launch_mr<TX, TY, kRowsPerPass>(fn, x, w, y, M, split_k(w.K), stream);
    else launch_mr<TX, TY, kWideRowsPerPass>(fn, x, w, y, M, split_k(w.K), stream);
}

}  // namespace

void linear_q5(const void *x, const Q5DeviceView &w, void *y, int64_t M, Act act, hipStream_t stream) {
    linear_q5(x, w, y, M, act, act, stream);
}

void linear_q5(const void *x, const Q5DeviceView &w, void *y, int64_t M, Act act, Act out_act, hipStream_t stream) {
    constexpr const char *fn = "linear_q5";
    constexpr int64_t kKAlign = 128;
    STRIX_CHECK(x != nullptr && y != nullptr && w.q != nullptr && w.scale != nullptr && w.minv != nullptr, fn,
                ": null pointer (x=", x, ", y=", y, ", q=", (const void *)w.q, ", scale=", w.scale, ", min=", w.minv, ")");
    STRIX_CHECK(w.G == 32 || w.G == 64 || w.G == 128, fn, ": group size ", w.G, " unsupported (32, 64 or 128)");
    STRIX_CHECK(M >= 1 && M <= kMaxM, fn, ": M (rows of x) = ", M, ", expected 1..", kMaxM);
    STRIX_CHECK(w.N >= 1 && w.N <= kMaxN, fn, ": N = ", w.N, ", expected 1..", kMaxN);
    STRIX_CHECK(w.K >= w.G && w.K <= kMaxK && w.K % w.G == 0 && w.K % kKAlign == 0, fn, ": K = ", w.K,
                " must be a positive multiple of group size ", w.G, " and of ", kKAlign, ", and <= ", kMaxK);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(w.q) % 16 == 0, fn, ": code array ", (const void *)w.q,
                " must be 16-byte aligned for vector loads");
    STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % 16 == 0, fn, ": x ", x, " must be 16-byte aligned for vector loads");
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
    STRIX_CHECK(out_act == act || (act == Act::BF16 && out_act == Act::F32), fn, ": output dtype ", act_name(out_act),
                " with input ", act_name(act), " unsupported (same dtype, or BF16 -> F32)");
    const size_t es = act == Act::F32 ? 4 : 2, oes = out_act == Act::F32 ? 4 : 2;
    const auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    const uintptr_t xe = xb + (uintptr_t)(M * w.K) * es, ye = yb + (uintptr_t)(M * w.N) * oes;
    STRIX_CHECK(ye <= xb || xe <= yb, fn, ": output y [", y, ", +", M * w.N * (int64_t)oes, " B) overlaps input x [",
                x, ", +", M * w.K * (int64_t)es, " B)");

    if (act == Act::F32) launch<float, float>(fn, static_cast<const float *>(x), w, static_cast<float *>(y), M, stream);
    else if (out_act == Act::F32)
        launch<uint16_t, float>(fn, static_cast<const uint16_t *>(x), w, static_cast<float *>(y), M, stream);
    else launch<uint16_t, uint16_t>(fn, static_cast<const uint16_t *>(x), w, static_cast<uint16_t *>(y), M, stream);
    STRIX_HIP_CHECK_LAUNCH(fn, " ", act_name(act), "->", act_name(out_act), " M=", M, " N=", w.N, " K=", w.K, " G=", w.G);
}

}  // namespace strix::kernels
