#include "kernels/vision.hpp"

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "kernels/wmma.hpp"

#include <cmath>
#include <cstdint>

namespace strix::kernels {

namespace {

// Shared helper shape (kernels/residual.hip does the same for its two ops).
__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

template <typename T> __device__ inline float load_as_f32(const T *p);
template <> __device__ inline float load_as_f32<float>(const float *p) { return *p; }
template <> __device__ inline float load_as_f32<uint16_t>(const uint16_t *p) { return bf16_bits_to_f32(*p); }
template <typename T> __device__ inline void store_f32(T *p, float v);
template <> __device__ inline void store_f32<float>(float *p, float v) { *p = v; }
template <> __device__ inline void store_f32<uint16_t>(uint16_t *p, float v) { *p = f32_to_bf16_bits(v); }

constexpr int kThreads = 256;

// ---------------------------------------------------------------- bias add
template <typename T>
__global__ void __launch_bounds__(kThreads) bias_add_kernel(T *y, const T *bias, int64_t cols, int64_t total) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (int64_t)gridDim.x * blockDim.x)
        store_f32(y + i, load_as_f32(y + i) + load_as_f32(bias + (i % cols)));
}

// ---------------------------------------------------------------- layernorm
// One block per row; FP32 stats (block reduce), write back after the reads of the row are done.
template <typename T>
__global__ void __launch_bounds__(kThreads) layernorm_kernel(const T *x, const float *g, const float *beta, T *out,
                                                             int64_t d, float eps) {
    const int64_t row = blockIdx.x;
    const T *xr = x + row * d;
    __shared__ float swarp[2][kThreads / 32];
    __shared__ float s_mean, s_inv;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarp = kThreads / 32;
    // Two-pass (mean, then central second moment) like torch's layer_norm: one-pass E[x^2] - mean^2
    // cancels on the ViT's outlier columns (values in the thousands).
    float sum = 0;
    for (int64_t k = threadIdx.x; k < d; k += kThreads) sum += load_as_f32(xr + k);
    for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(~0ull, sum, o);
    if (lane == 0) swarp[0][wid] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0;
        for (int i = 0; i < nwarp; ++i) m += swarp[0][i];
        s_mean = m / (float)d;
    }
    __syncthreads();
    const float mean = s_mean;
    float sq = 0;
    for (int64_t k = threadIdx.x; k < d; k += kThreads) {
        const float c = load_as_f32(xr + k) - mean;
        sq += c * c;
    }
    for (int o = 16; o; o >>= 1) sq += __shfl_down_sync(~0ull, sq, o);
    if (lane == 0) swarp[1][wid] = sq;
    __syncthreads();
    if (threadIdx.x == 0) {
        float q = 0;
        for (int i = 0; i < nwarp; ++i) q += swarp[1][i];
        s_inv = rsqrtf(q / (float)d + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    T *orow = out + row * d;
    for (int64_t k = threadIdx.x; k < d; k += kThreads)
        store_f32(orow + k, (load_as_f32(xr + k) - mean) * inv * g[k] + beta[k]);
}

// ---------------------------------------------------------------- gelu
__device__ inline float gelu_one(float x, bool tanh_form) {
    if (tanh_form) return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

template <typename T>
__global__ void __launch_bounds__(kThreads) gelu_kernel(T *x, int64_t n, bool tanh_form) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        store_f32(x + i, gelu_one(load_as_f32(x + i), tanh_form));
}

// ---------------------------------------------------------------- position-table resample add
template <typename T>
__global__ void __launch_bounds__(kThreads) pos_add_kernel(const T *in, const uint16_t *table, const int32_t *idx,
                                                           const float *w, T *out, int64_t d, int64_t taps) {
    const int64_t row = blockIdx.x;
    const T *xr = in + row * d;
    const int32_t *ir = idx + row * taps;
    const float *wr = w + row * taps;
    float wt[4];
    const uint16_t *tp[4];
    const int nt = (int)taps;
    for (int t = 0; t < nt && t < 4; ++t) { wt[t] = wr[t]; tp[t] = table + (int64_t)ir[t] * d; }
    T *orow = out + row * d;
    for (int64_t k = threadIdx.x; k < d; k += kThreads) {
        float acc = load_as_f32(xr + k);
        for (int t = 0; t < nt && t < 4; ++t) acc += wt[t] * bf16_bits_to_f32(tp[t][k]);
        store_f32(orow + k, acc);
    }
}

// ---------------------------------------------------------------- rope (q, k slices of fused qkv)
// Row -> (h, w) from the grid in merge-block order (the tower's patch order).
__device__ inline void grid_hw(int64_t r, int64_t grid_h, int64_t grid_w, int64_t merge, int64_t &h, int64_t &w) {
    const int64_t blocks_w = grid_w / merge, m2 = merge * merge;
    const int64_t blk = r / m2, sub = r % m2;
    h = (blk / blocks_w) * merge + sub / merge;
    w = (blk % blocks_w) * merge + sub % merge;
}

template <typename T>
__global__ void __launch_bounds__(kThreads) rope_qk_kernel(T *qkv, int64_t n, int64_t heads, int64_t hd,
                                                           int64_t grid_h, int64_t grid_w, int64_t merge,
                                                           float theta) {
    const int64_t row = blockIdx.x;
    if (row >= n) return;
    int64_t h, w;
    grid_hw(row, grid_h, grid_w, merge, h, w);
    const int64_t spatial = hd / 2, nf = spatial / 2;  // 36, 18
    const float step = 1.0f / spatial;
    T *qrow = qkv + row * (3 * heads * hd);
    // One thread owns a full rotate_half PAIR (d, d + spatial) - never a single element - so no thread can
    // read a partner another thread already rotated. q columns [0, heads*hd), k columns [heads*hd, 2*).
    for (int64_t e = threadIdx.x; e < 2 * heads * spatial; e += kThreads) {
        const int64_t d = e % spatial, head = e / spatial;  // head counts q|k slices (2 * heads)
        const int64_t col = head * hd + d;                 // contiguous across the q and k slices
        const int64_t fi = d >= 2 * nf ? d - 2 * nf : d;  // [h | w | h | w]: the second half mirrors
        const bool use_w = fi >= nf && fi < 2 * nf;
        const float pos = (float)(use_w ? w : h);
        const float freq = powf(theta, -(float)(2 * (fi % nf)) * step);
        const float ang = pos * freq;
        const float c = cosf(ang), s = sinf(ang);
        const float a = load_as_f32(qrow + col), b = load_as_f32(qrow + col + spatial);
        store_f32(qrow + col, a * c - b * s);
        store_f32(qrow + col + spatial, b * c + a * s);
    }
}

// ---------------------------------------------------------------- GEMM + bias (single-rounding epilogue)
// One thread per output element y[m, n] = dot(x[m, :], W[n, :]) + bias[n]: dev-stage simple form (the ViT
// runs once per image; the x row is shared by the block and the W rows are L2-resident across m). The
// tuned path can later swap under the same contract.
template <typename T>
__global__ void __launch_bounds__(kThreads) linear_bias_kernel(const T *x, const uint16_t *w, const T *bias, T *y,
                                                              int64_t M, int64_t N, int64_t K) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    const int64_t m = idx / N, n = idx % N;
    const T *xr = x + m * K;
    const uint16_t *wr = w + n * K;
    float acc = 0.f;
    for (int64_t k = 0; k < K; ++k) acc += load_as_f32(xr + k) * bf16_bits_to_f32(wr[k]);
    store_f32(y + idx, acc + load_as_f32(bias + n));
}

// ---------------------------------------------------------------- GEMM + bias, BF16 activations (WMMA)
// y[m, n] = dot(x[m, :], W[n, :]) + bias[n] on the matrix cores: block 64 x-rows by 128 W-rows, 8 waves as 2 x 4,
// each wave 32 x 32 = 2 x 2 16x16 tiles. A K slice of 64 of both tiles is staged in LDS (rows padded to 72 so the
// 16-byte fragment reads spread over banks); x and W are both BF16, so the fragments are the stored values - no
// extra rounding, FP32 accumulation. Rows past M / N and K past the end are zero-filled; the host requires
// K % 16 == 0 (every tower K is) and 16-byte aligned x / w, so a 16-wide K run is all in or all out. The bias is
// added to the FP32 accumulator before the one rounding, as in linear_bias_kernel.
constexpr int kGemmBM = 64, kGemmBN = 128, kGemmBK = 64, kGemmLd = kGemmBK + 8;

__global__ void __launch_bounds__(kThreads) linear_bias_wmma_kernel(const uint16_t *x, const uint16_t *w,
                                                                   const uint16_t *bias, uint16_t *y, int64_t M,
                                                                   int64_t N, int64_t K) {
    __shared__ __attribute__((aligned(16))) uint16_t xs[kGemmBM * kGemmLd];
    __shared__ __attribute__((aligned(16))) uint16_t ws[kGemmBN * kGemmLd];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int wm = wave / 4, wn = wave % 4;  // wave's 32 x 32 sub-tile
    const int64_t m0 = (int64_t)blockIdx.y * kGemmBM, n0 = (int64_t)blockIdx.x * kGemmBN;
    wmma_f32x8 acc[2][2] = {};
    for (int64_t k0 = 0; k0 < K; k0 += kGemmBK) {
        wmma_f32x8 part[2][2] = {};  // this K slice's sum, added to acc with IEEE FP32 adds (see below)
        // Stage: 8 BF16 (one uint4) per load; x tile 64 x 64 = 512 loads, W tile 128 x 64 = 1024 loads.
        for (int i = tid; i < (kGemmBM + kGemmBN) * (kGemmBK / 8); i += kThreads) {
            const int r = i / (kGemmBK / 8), c = (i % (kGemmBK / 8)) * 8;
            const bool is_x = r < kGemmBM;
            const int rr = is_x ? r : r - kGemmBM;
            const int64_t grow = (is_x ? m0 : n0) + rr, gk = k0 + c;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (grow < (is_x ? M : N) && gk < K)
                v = *reinterpret_cast<const uint4 *>((is_x ? x : w) + grow * K + gk);
            *reinterpret_cast<uint4 *>((is_x ? xs : ws) + rr * kGemmLd + c) = v;
        }
        __syncthreads();
        const int kn = (int)min<int64_t>(kGemmBK, K - k0);  // a multiple of 16
        for (int kk = 0; kk < kn; kk += 16) {
            wmma_bf16x16 a[2], b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) a[i] = wmma_load_rows_a16(xs + (wm * 32 + i * 16) * kGemmLd + kk, kGemmLd);
#pragma unroll
            for (int j = 0; j < 2; ++j) b[j] = wmma_load_rows_a16(ws + (wn * 32 + j * 16) * kGemmLd + kk, kGemmLd);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) part[i][j] += wmma_bf16(a[i], b[j], wmma_f32x8{});
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) acc[i][j] += part[i][j];
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int64_t n = n0 + wn * 32 + j * 16 + lane % 16;
            if (n >= N) continue;
            const float bn = bf16_bits_to_f32(bias[n]);
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int64_t m = m0 + wm * 32 + i * 16 + wmma_c_row(v);
                if (m < M) y[m * N + n] = f32_to_bf16_bits(acc[i][j][v] + bn);
            }
        }
}

// ---------------------------------------------------------------- merge regroup (pure permutation)
template <typename T>
__global__ void __launch_bounds__(kThreads) merge_regroup_kernel(const T *in, T *out, int64_t d, int64_t m2,
                                                                int64_t total) {
    for (int64_t o = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; o < total; o += (int64_t)gridDim.x * blockDim.x) {
        const int64_t md = m2 * d;
        const int64_t u = o / md, rem = o % md, slot = rem / d, c = rem % d;
        store_f32(out + o, load_as_f32(in + (u * m2 + slot) * d + c));
    }
}
// ---------------------------------------------------------------- embedding-row scatter (pure copy)
// One block per image row: copies*d consecutive elements (stream copy c at offset c * d) of the merged
// row, written at the target row's stride out_ts (>= copies * d).
template <typename T>
__global__ void __launch_bounds__(kThreads) embedding_scatter_kernel(T *x, const int32_t *row_dev, const T *merged,
                                                                    int64_t d, int64_t out_ts, int64_t width) {
    const int64_t r = blockIdx.x;
    T *dst = x + (int64_t)row_dev[r] * out_ts;
    const T *src = merged + r * d;
    for (int64_t e = threadIdx.x; e < width; e += kThreads) store_f32(dst + e, load_as_f32(src + e % d));
}

// ---------------------------------------------------------------- attention, BF16 (WMMA, online softmax)
// Block (128 query rows, head), 8 waves of 16 rows; keys in tiles of 64. Per tile: K [64 keys x hd] and V transposed
// [hd x 64 keys] staged in LDS (head dim zero-padded to kAttnD = 80, 5 WMMA K steps), S = Q K^T on the matrix cores
// (fresh accumulator per 16-wide step, summed with FP32 adds - as the GEMM), scaled, keys past n masked, running max
// / sum per row in FP32 with exp2 on log2e-prescaled scores, P rounded to BF16 (via LDS, accumulator -> A-fragment layout) for O += P V.
// Accumulator element v of lane l is row 2v + l / 16, column l % 16, so a row's 16 columns sit in one half-wave and
// its max / sum reduce over xor 1, 2, 4, 8. The host requires hd % 8 == 0, hd <= kAttnD, 16-byte aligned qkv rows.
constexpr int kAttnRows = 128, kAttnKeys = 64, kAttnD = 80, kAttnLd = kAttnD + 8, kAttnKLd = kAttnKeys + 8;
constexpr int kAttnThreads = 256;  // 8 waves x 16 rows

__global__ void __launch_bounds__(kAttnThreads) attn_wmma_kernel(const uint16_t *qkv, uint16_t *out, int64_t n,
                                                                 int64_t heads, int64_t hd, float scale) {
    __shared__ __attribute__((aligned(16))) uint16_t qs[kAttnRows * kAttnLd];
    __shared__ __attribute__((aligned(16))) uint16_t ks[kAttnKeys * kAttnLd];
    __shared__ __attribute__((aligned(16))) uint16_t vt[kAttnD * kAttnKLd];
    const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
    const int64_t q0 = (int64_t)blockIdx.x * kAttnRows, hh = blockIdx.y, stride = 3 * heads * hd;
    const int dv = (int)hd / 8;  // uint4 loads per row
    // Scores in the log2 domain: exp(x * scale) = exp2(x * scale * log2 e) - the hardware exp2, ~1 ulp.
    const float scale_log2e = scale * 1.4426950408889634f;
    // Q rows (zero past n and past hd).
    for (int i = tid; i < kAttnRows * (kAttnD / 8); i += kAttnThreads) {
        const int r = i / (kAttnD / 8), c = i % (kAttnD / 8);
        uint4 v = make_uint4(0, 0, 0, 0);
        if (q0 + r < n && c < dv) v = *reinterpret_cast<const uint4 *>(qkv + (q0 + r) * stride + hh * hd + c * 8);
        *reinterpret_cast<uint4 *>(qs + r * kAttnLd + c * 8) = v;
    }
    __syncthreads();
    wmma_bf16x16 qa[kAttnD / 16];
#pragma unroll
    for (int d = 0; d < kAttnD / 16; ++d) qa[d] = wmma_load_rows_a16(qs + wave * 16 * kAttnLd + d * 16, kAttnLd);
    wmma_f32x8 o[kAttnD / 16] = {};
    float mrow[8], lrow[8];
#pragma unroll
    for (int v = 0; v < 8; ++v) mrow[v] = -INFINITY, lrow[v] = 0.f;
    // P reuses the Q staging rows: every wave holds its Q fragments in registers before the loop's first barrier.
    static_assert(kAttnThreads / 32 * 16 * kAttnKLd <= kAttnRows * kAttnLd, "P must fit in the Q staging area");
    uint16_t *pw = qs + wave * 16 * kAttnKLd;
    for (int64_t k0 = 0; k0 < n; k0 += kAttnKeys) {
        __syncthreads();  // the previous tile's K / V reads are done
        for (int i = tid; i < kAttnKeys * (kAttnD / 8); i += kAttnThreads) {
            const int r = i / (kAttnD / 8), c = i % (kAttnD / 8);
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (k0 + r < n && c < dv) {
                const uint16_t *row = qkv + (k0 + r) * stride + hh * hd + c * 8;
                kv = *reinterpret_cast<const uint4 *>(row + heads * hd);
                vv = *reinterpret_cast<const uint4 *>(row + 2 * heads * hd);
            }
            *reinterpret_cast<uint4 *>(ks + r * kAttnLd + c * 8) = kv;
            const uint16_t *ve = reinterpret_cast<const uint16_t *>(&vv);
#pragma unroll
            for (int e = 0; e < 8; ++e) vt[(c * 8 + e) * kAttnKLd + r] = ve[e];
        }
        __syncthreads();
        // S for this wave's 16 rows x 64 keys: 4 tiles of 16 keys.
        wmma_f32x8 sc[kAttnKeys / 16];
#pragma unroll
        for (int t = 0; t < kAttnKeys / 16; ++t) {
            sc[t] = wmma_f32x8{};
#pragma unroll
            for (int d = 0; d < kAttnD / 16; ++d)
                sc[t] = wmma_bf16(qa[d], wmma_load_rows_a16(ks + t * 16 * kAttnLd + d * 16, kAttnLd), sc[t]);  // K = 80: short chain
            const bool masked = k0 + t * 16 + lane % 16 >= n;
#pragma unroll
            for (int v = 0; v < 8; ++v) sc[t][v] = masked ? -INFINITY : sc[t][v] * scale_log2e;
        }
        // Online softmax per row (v), over the half-wave's 16 lanes.
        float alpha[8];
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            float mx = fmaxf(fmaxf(sc[0][v], sc[1][v]), fmaxf(sc[2][v], sc[3][v]));
#pragma unroll
            for (int o2 = 1; o2 < 16; o2 <<= 1) mx = fmaxf(mx, __shfl_xor(mx, o2, 32));
            const float mnew = fmaxf(mrow[v], mx);
            alpha[v] = exp2f(mrow[v] - mnew);  // exp2(-inf) = 0 on the first tile
            float sum = 0.f;
#pragma unroll
            for (int t = 0; t < kAttnKeys / 16; ++t) {
                const float pv = exp2f(sc[t][v] - mnew);
                sum += pv;
                pw[wmma_c_row(v) * kAttnKLd + t * 16 + lane % 16] = f32_to_bf16_bits(pv);
            }
#pragma unroll
            for (int o2 = 1; o2 < 16; o2 <<= 1) sum += __shfl_xor(sum, o2, 32);
            lrow[v] = lrow[v] * alpha[v] + sum;
            mrow[v] = mnew;
        }
#pragma unroll
        for (int d = 0; d < kAttnD / 16; ++d)
#pragma unroll
            for (int v = 0; v < 8; ++v) o[d][v] *= alpha[v];
        __syncthreads();  // P written (each wave reads only its own rows, but the block shares the barrier)
        // P V for this tile chained over its 64 keys on the matrix core, then added to O with FP32 adds (the long
        // accumulation over all keys stays IEEE, as the GEMM's).
        wmma_f32x8 op[kAttnD / 16] = {};
#pragma unroll
        for (int t = 0; t < kAttnKeys / 16; ++t) {
            const wmma_bf16x16 pa = wmma_load_rows_a16(pw + t * 16, kAttnKLd);
#pragma unroll
            for (int d = 0; d < kAttnD / 16; ++d)
                op[d] = wmma_bf16(pa, wmma_load_rows_a16(vt + d * 16 * kAttnKLd + t * 16, kAttnKLd), op[d]);
        }
#pragma unroll
        for (int d = 0; d < kAttnD / 16; ++d) o[d] += op[d];
    }
#pragma unroll
    for (int d = 0; d < kAttnD / 16; ++d) {
        const int64_t col = d * 16 + lane % 16;
        if (col >= hd) continue;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int64_t row = q0 + wave * 16 + wmma_c_row(v);
            if (row < n) out[row * heads * hd + hh * hd + col] = f32_to_bf16_bits(o[d][v] / lrow[v]);
        }
    }
}

// ---------------------------------------------------------------- attention (one segment, dense, non-causal)
// Block (row, head): 72-wide q in shared, scores over all keys in shared, softmax, PV.
template <typename T>
__global__ void __launch_bounds__(kThreads) attn_kernel(const T *qkv, T *out, int64_t n, int64_t heads, int64_t hd,
                                                        float scale) {
    const int64_t row = blockIdx.x, hh = blockIdx.y;
    const int64_t stride = 3 * heads * hd;
    const T *qbase = qkv + hh * hd, *vbase = qkv + (2 * heads * hd) + hh * hd;
    extern __shared__ float smem[];               // [n] scores, then [hd] q
    float *scores = smem;
    float *qs = smem + n;
    for (int64_t d = threadIdx.x; d < hd; d += kThreads) qs[d] = load_as_f32(qbase + row * stride + d);
    __syncthreads();
    float local_max = -INFINITY;
    for (int64_t j = threadIdx.x; j < n; j += kThreads) {
        const T *kj = qkv + heads * hd + hh * hd + j * stride;
        float dot = 0.f;
        for (int64_t d = 0; d < hd; ++d) dot += qs[d] * load_as_f32(kj + d);
        scores[j] = dot * scale;
        local_max = fmaxf(local_max, scores[j]);
    }
    // block max
    __shared__ float sred[kThreads / 32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nwarp = kThreads / 32;
    for (int o = 16; o; o >>= 1) local_max = fmaxf(local_max, __shfl_down_sync(~0ull, local_max, o));
    if (lane == 0) sred[wid] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int i = 0; i < nwarp; ++i) m = fmaxf(m, sred[i]);
        sred[0] = m;
    }
    __syncthreads();
    const float mx = sred[0];
    float lsum = 0.f;
    for (int64_t j = threadIdx.x; j < n; j += kThreads) {
        scores[j] = expf(scores[j] - mx);  // precise expf: the dev-stage reference parity beats the fast one
        lsum += scores[j];
    }
    float sm = lsum;
    for (int o = 16; o; o >>= 1) sm += __shfl_down_sync(~0ull, sm, o);
    __shared__ float sred2[kThreads / 32];
    if (lane == 0) sred2[wid] = sm;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < nwarp; ++i) s += sred2[i];
        sred2[0] = s;
    }
    __syncthreads();
    const float inv = 1.0f / sred2[0];
    // PV: one accumulator per output dim per thread group; threads cover hd columns, stride over keys.
    for (int64_t d = threadIdx.x; d < hd; d += kThreads) {
        float acc = 0.f;
        for (int64_t j = 0; j < n; ++j) acc += scores[j] * inv * load_as_f32(vbase + j * stride + d);
        store_f32(out + row * (heads * hd) + hh * hd + d, acc);
    }
}

}  // namespace

// ---------------------------------------------------------------- host launchers

namespace {

constexpr int64_t kElemsCap = 1ll << 40;

void check_act(Act act, const char *fn) {
    STRIX_CHECK(act == Act::F32 || act == Act::BF16, fn, ": unsupported activation dtype ", (int)act);
}
size_t elem_size(Act act) { return act == Act::F32 ? 4 : 2; }
void check_aligned(const void *p, size_t es, const char *fn, const char *what) {
    STRIX_CHECK(p != nullptr, fn, ": null ", what);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(p) % es == 0, fn, ": ", what, " = ", p, " not ", es, "-byte aligned");
}
dim3 grid_for(int64_t units, unsigned block = kThreads) {
    const int64_t blocks = (units + block - 1) / block;
    STRIX_CHECK(blocks <= 2147483647ll, "kernel grid units ", units, " exceed the grid limit");
    return dim3((unsigned)blocks);
}

// For the one-block-per-row kernels (layernorm, pos_add, rope): grid.x = rows.
dim3 grid_rows(int64_t rows) {
    STRIX_CHECK(rows <= 2147483647ll, "kernel rows ", rows, " exceed the grid limit");
    return dim3((unsigned)rows);
}

}  // namespace

void vision_bias_add(void *y, const void *bias, int64_t rows, int64_t cols, Act act, hipStream_t stream) {
    const char *fn = "vision_bias_add";
    STRIX_CHECK(rows >= 1 && cols >= 1 && rows * cols <= kElemsCap, fn, ": rows x cols = ", rows, " x ", cols);
    check_act(act, fn);
    const size_t es = elem_size(act);
    check_aligned(y, es, fn, "y");
    check_aligned(bias, es, fn, "bias");
    const int64_t total = rows * cols;
    if (act == Act::F32)
        bias_add_kernel<float><<<grid_for(total), kThreads, 0, stream>>>((float *)y, (const float *)bias, cols, total);
    else
        bias_add_kernel<uint16_t><<<grid_for(total), kThreads, 0, stream>>>((uint16_t *)y, (const uint16_t *)bias,
                                                                            cols, total);
    STRIX_HIP_CHECK_LAUNCH(fn, " rows=", rows, " cols=", cols, " ", act_name(act));
}

void vision_layernorm(const void *x, const float *g, const float *beta, void *out, int64_t rows, int64_t d,
                      float eps, Act act, hipStream_t stream) {
    const char *fn = "vision_layernorm";
    STRIX_CHECK(rows >= 1 && d >= 1, fn, ": rows = ", rows, ", d = ", d);
    STRIX_CHECK(std::isfinite(eps) && eps > 0.f, fn, ": eps = ", eps);
    check_act(act, fn);
    check_aligned(x, elem_size(act), fn, "x");
    check_aligned(out, elem_size(act), fn, "out");
    check_aligned(g, 4, fn, "g");
    check_aligned(beta, 4, fn, "beta");
    if (act == Act::F32)
        layernorm_kernel<float><<<grid_rows(rows), kThreads, 0, stream>>>((const float *)x, g, beta, (float *)out, d,
                                                                         eps);
    else
        layernorm_kernel<uint16_t><<<grid_rows(rows), kThreads, 0, stream>>>((const uint16_t *)x, g, beta,
                                                                            (uint16_t *)out, d, eps);
    STRIX_HIP_CHECK_LAUNCH(fn, " rows=", rows, " d=", d, " ", act_name(act));
}

const char *vision_gelu_name(VisionGelu g) { return g == VisionGelu::Tanh ? "gelu_tanh" : "gelu_erf"; }

void vision_gelu(void *x, int64_t n, VisionGelu gelu, Act act, hipStream_t stream) {
    const char *fn = "vision_gelu";
    STRIX_CHECK(n >= 1 && n <= kElemsCap, fn, ": n = ", n);
    check_act(act, fn);
    check_aligned(x, elem_size(act), fn, "x");
    const bool tanh_form = gelu == VisionGelu::Tanh;
    if (act == Act::F32)
        gelu_kernel<float><<<grid_for(n), kThreads, 0, stream>>>((float *)x, n, tanh_form);
    else
        gelu_kernel<uint16_t><<<grid_for(n), kThreads, 0, stream>>>((uint16_t *)x, n, tanh_form);
    STRIX_HIP_CHECK_LAUNCH(fn, " n=", n, " ", act_name(act));
}

void vision_pos_add(const void *in, const uint16_t *table, const int32_t *idx, const float *w, void *out,
                    int64_t rows, int64_t d, int64_t taps, Act act, hipStream_t stream) {
    const char *fn = "vision_pos_add";
    STRIX_CHECK(rows >= 1 && d >= 1, fn, ": rows = ", rows, ", d = ", d);
    STRIX_CHECK(taps >= 1 && taps <= 4, fn, ": taps = ", taps, ", expected 1..4 (bilinear)");
    check_act(act, fn);
    check_aligned(in, elem_size(act), fn, "in");
    check_aligned(out, elem_size(act), fn, "out");
    check_aligned(table, 2, fn, "table");
    check_aligned(idx, 4, fn, "idx");
    check_aligned(w, 4, fn, "w");
    if (act == Act::F32)
        pos_add_kernel<float><<<grid_rows(rows), kThreads, 0, stream>>>((const float *)in, table, idx, w,
                                                                        (float *)out, d, (int)taps);
    else
        pos_add_kernel<uint16_t><<<grid_rows(rows), kThreads, 0, stream>>>((const uint16_t *)in, table, idx, w,
                                                                           (uint16_t *)out, d, (int)taps);
    STRIX_HIP_CHECK_LAUNCH(fn, " rows=", rows, " d=", d, " ", act_name(act));
}

void vision_rope_qk(void *qkv, int64_t n, int64_t heads, int64_t hd, int64_t grid_h, int64_t grid_w,
                    int64_t merge, float theta, Act act, hipStream_t stream) {
    const char *fn = "vision_rope_qk";
    STRIX_CHECK(n >= 1 && heads >= 1, fn, ": n = ", n, ", heads = ", heads);
    STRIX_CHECK(hd >= 2 && hd % 2 == 0 && (hd / 2) % 2 == 0, fn, ": head_dim ", hd,
                " must be even with an even half (freq pairs)");
    STRIX_CHECK(grid_h >= 1 && grid_w >= 1 && merge >= 1 && grid_h % merge == 0 && grid_w % merge == 0,
                fn, ": grid ", grid_h, "x", grid_w, " merge ", merge);
    STRIX_CHECK(n == grid_h * grid_w, fn, ": rows ", n, " != grid cells ", grid_h * grid_w);
    STRIX_CHECK(std::isfinite(theta) && theta > 0.f, fn, ": theta = ", theta);
    check_act(act, fn);
    check_aligned(qkv, elem_size(act), fn, "qkv");
    if (act == Act::F32)
        rope_qk_kernel<float><<<grid_rows(n), kThreads, 0, stream>>>((float *)qkv, n, heads, hd, grid_h, grid_w,
                                                                    merge, theta);
    else
        rope_qk_kernel<uint16_t><<<grid_rows(n), kThreads, 0, stream>>>((uint16_t *)qkv, n, heads, hd, grid_h,
                                                                       grid_w, merge, theta);
    STRIX_HIP_CHECK_LAUNCH(fn, " n=", n, " ", act_name(act));
}

void vision_attn(const void *qkv, void *out, int64_t n, int64_t heads, int64_t hd, Act act, hipStream_t stream) {
    const char *fn = "vision_attn";
    STRIX_CHECK(n >= 1 && n <= kMaxVisionSeq, fn, ": n = ", n, ", expected 1..", kMaxVisionSeq,
                " (per-segment length; one image = one segment)");
    STRIX_CHECK(heads >= 1 && hd >= 1 && heads * hd <= kElemsCap, fn, ": heads x head_dim = ", heads, " x ", hd);
    check_act(act, fn);
    const size_t es = elem_size(act);
    check_aligned(qkv, es, fn, "qkv");
    check_aligned(out, es, fn, "out");
    const auto qb = reinterpret_cast<uintptr_t>(qkv), ob = reinterpret_cast<uintptr_t>(out);
    STRIX_CHECK(ob >= qb + (uintptr_t)(n * 3 * heads * hd) * es || qb >= ob + (uintptr_t)(n * heads * hd) * es,
                fn, ": out [", out, ", +", n * heads * hd * (int64_t)es,
                " B) overlaps qkv [", qkv, ", +", n * 3 * heads * hd * (int64_t)es,
                " B) - rows are read across all outputs (two buffers)");
    const float scale = (float)(1.0 / std::sqrt((double)hd));
    dim3 grid((unsigned)n, (unsigned)heads);
    const size_t smem = sizeof(float) * ((size_t)n + hd);
    STRIX_CHECK(smem <= 64 * 1024, fn, ": shared memory ", smem, " exceeds the 64 KiB static cap");
    if (act == Act::F32)
        attn_kernel<float><<<grid, kThreads, smem, stream>>>((const float *)qkv, (float *)out, n, heads, hd, scale);
    else {
        // The WMMA path's preconditions: whole 16-byte loads of a head's columns, head dim within the padded tile.
        STRIX_CHECK(hd % 8 == 0 && hd <= kAttnD, fn, ": BF16 needs head_dim % 8 == 0 and <= ", kAttnD, ", got ", hd);
        STRIX_CHECK(reinterpret_cast<uintptr_t>(qkv) % 16 == 0, fn, ": BF16 needs 16-byte aligned qkv, got ", qkv);
        const dim3 g((unsigned)((n + kAttnRows - 1) / kAttnRows), (unsigned)heads);
        attn_wmma_kernel<<<g, kAttnThreads, 0, stream>>>((const uint16_t *)qkv, (uint16_t *)out, n, heads, hd, scale);
    }
    STRIX_HIP_CHECK_LAUNCH(fn, " n=", n, " heads=", heads, " ", act_name(act));
}

void vision_linear_bias(const void *x, const uint16_t *w, const void *bias, void *y, int64_t M, int64_t N,
                        int64_t K, Act act, hipStream_t stream) {
    const char *fn = "vision_linear_bias";
    STRIX_CHECK(M >= 1 && N >= 1 && K >= 1, fn, ": M,N,K = ", M, ", ", N, ", ", K);
    check_act(act, fn);
    const size_t es = elem_size(act);
    check_aligned(x, es, fn, "x");
    check_aligned(w, 2, fn, "w");
    check_aligned(bias, es, fn, "bias");
    check_aligned(y, es, fn, "y");
    const auto xb = reinterpret_cast<uintptr_t>(x), yb = reinterpret_cast<uintptr_t>(y);
    STRIX_CHECK(yb >= xb + (uintptr_t)(M * K) * es || xb >= yb + (uintptr_t)(M * N) * es,
                fn, ": y overlaps x (in-place isn't supported)");
    const int64_t total = M * N;
    if (act == Act::F32)
        linear_bias_kernel<float><<<grid_for(total), kThreads, 0, stream>>>((const float *)x, w,
                                                                            (const float *)bias, (float *)y, M, N, K);
    else {
        // The WMMA path's preconditions: whole 16-wide K runs and 16-byte row loads.
        STRIX_CHECK(K % 16 == 0, fn, ": BF16 needs K % 16 == 0, got K = ", K);
        STRIX_CHECK(reinterpret_cast<uintptr_t>(x) % 16 == 0 && reinterpret_cast<uintptr_t>(w) % 16 == 0, fn,
                    ": BF16 needs 16-byte aligned x and w, got x = ", x, ", w = ", static_cast<const void *>(w));
        STRIX_CHECK((N + kGemmBN - 1) / kGemmBN <= INT32_MAX && (M + kGemmBM - 1) / kGemmBM <= 65535, fn,
                    ": grid too large for M = ", M, ", N = ", N);
        const dim3 grid((unsigned)((N + kGemmBN - 1) / kGemmBN), (unsigned)((M + kGemmBM - 1) / kGemmBM));
        linear_bias_wmma_kernel<<<grid, kThreads, 0, stream>>>((const uint16_t *)x, w, (const uint16_t *)bias,
                                                               (uint16_t *)y, M, N, K);
    }
    STRIX_HIP_CHECK_LAUNCH(fn, " M=", M, " N=", N, " K=", K, " ", act_name(act));
}

void vision_merge_regroup(const void *in, void *out, int64_t rows, int64_t d, int64_t merge, Act act,
                          hipStream_t stream) {
    const char *fn = "vision_merge_regroup";
    const int64_t m2 = merge * merge;
    STRIX_CHECK(rows >= m2 && rows % m2 == 0 && d >= 1, fn, ": rows = ", rows, ", merge^2 = ", m2, ", d = ", d);
    check_act(act, fn);
    const size_t es = elem_size(act);
    check_aligned(in, es, fn, "in");
    check_aligned(out, es, fn, "out");
    const int64_t total = rows * d;
    if (act == Act::F32)
        merge_regroup_kernel<float><<<grid_for(total), kThreads, 0, stream>>>((const float *)in, (float *)out, d,
                                                                              m2, total);
    else
        merge_regroup_kernel<uint16_t><<<grid_for(total), kThreads, 0, stream>>>((const uint16_t *)in,
                                                                                 (uint16_t *)out, d, m2, total);
    STRIX_HIP_CHECK_LAUNCH(fn, " rows=", rows, " d=", d, " ", act_name(act));
}

void vision_embedding_scatter(void *x, const int32_t *row_dev, int64_t rows, const void *merged, int64_t d,
                              int64_t total_rows, int64_t copies, int64_t out_ts, Act act, hipStream_t stream) {
    const char *fn = "vision_embedding_scatter";
    STRIX_CHECK(rows >= 1 && d >= 1, fn, ": rows = ", rows, ", d = ", d);
    STRIX_CHECK(copies >= 1 && out_ts >= copies * d, fn, ": copies = ", copies, ", out_ts = ", out_ts,
                ", expected out_ts >= copies * d = ", copies * d);
    STRIX_CHECK(total_rows >= 1, fn, ": total_rows = ", total_rows);
    check_act(act, fn);
    const size_t es = elem_size(act);
    check_aligned(x, es, fn, "x");
    check_aligned(row_dev, 4, fn, "row_dev");
    check_aligned(merged, es, fn, "merged");
    // The row bounds live on the device - the caller (the session's image forward, which derives the
    // indices from its own host-side id list) validates them before the upload.
    if (act == Act::F32)
        embedding_scatter_kernel<float><<<grid_rows(rows), kThreads, 0, stream>>>(
            (float *)x, row_dev, (const float *)merged, d, out_ts, copies * d);
    else
        embedding_scatter_kernel<uint16_t><<<grid_rows(rows), kThreads, 0, stream>>>(
            (uint16_t *)x, row_dev, (const uint16_t *)merged, d, out_ts, copies * d);
    STRIX_HIP_CHECK_LAUNCH(fn, " rows=", rows, " d=", d, " ", act_name(act));
}

}  // namespace strix::kernels
