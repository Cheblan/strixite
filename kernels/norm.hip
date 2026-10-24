#include "kernels/norm.hpp"

#include "common/hip_check.hpp"
#include "kernels/device_info.hpp"

#include <cmath>
#include <cstdint>

namespace strix::kernels {

namespace {

constexpr int kThreads = 256;           // one block per row; LDS tree reduction, no wave-size assumption
constexpr int64_t kMaxRowLen = 1 << 20; // far above any real norm width; catches a rows/d mix-up
constexpr int64_t kMaxRows = 0x7fffffff;
constexpr int64_t kMaxGroups = 64;      // qwen4_exp has 4; catches a groups/d or tokens/groups mix-up

// BF16 is carried as raw bits. RNE rounding, NaN kept a NaN - same as the
// host-side round_to_bf16 the tests use.
__device__ inline float bf16_bits_to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

template <typename T> __device__ inline float load(const T *p, int64_t i);
template <> __device__ inline float load<float>(const float *p, int64_t i) { return p[i]; }
template <> __device__ inline float load<uint16_t>(const uint16_t *p, int64_t i) { return bf16_bits_to_f32(p[i]); }

template <typename T> __device__ inline void store(T *p, int64_t i, float v);
template <> __device__ inline void store<float>(float *p, int64_t i, float v) { p[i] = v; }
template <> __device__ inline void store<uint16_t>(uint16_t *p, int64_t i, float v) { p[i] = f32_to_bf16_bits(v); }

// Round through the activation dtype (identity for F32).
template <typename T> __device__ inline float round_act(float v);
template <> __device__ inline float round_act<float>(float v) { return v; }
template <> __device__ inline float round_act<uint16_t>(float v) { return bf16_bits_to_f32(f32_to_bf16_bits(v)); }

// 1 / sqrt(mean(x^2) + eps) for this block's row; all threads get the result.
template <typename T>
__device__ inline float row_inv_rms(const T *xr, int64_t d, float eps, float *red) {
    float ss = 0.f;
    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
        float v = load(xr, j);
        ss += v * v;
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    // 1/sqrtf rather than rsqrtf: sqrtf/division are correctly rounded by default in HIP,
    // rsqrtf is an approximation.
    return 1.0f / sqrtf(red[0] / (float)d + eps);
}

// One block per d-wide row. Row r uses weight slice r % groups (groups = 1: plain rmsnorm_zc; grouped: the rows
// are a [tokens, groups * d] tensor's slices, so consecutive rows walk the groups).
template <typename T>
__global__ void __launch_bounds__(kThreads)
    rmsnorm_zc_kernel(const T *x, const float *w, T *out, int64_t d, int groups, float eps) {
    __shared__ float red[kThreads];
    const T *xr = x + (int64_t)blockIdx.x * d;
    T *orow = out + (int64_t)blockIdx.x * d;
    const float *wr = w + (int64_t)(blockIdx.x % (unsigned)groups) * d;
    float inv = row_inv_rms(xr, d, eps, red);
    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) store(orow, j, (load(xr, j) * inv) * (1.0f + wr[j]));
}

void launch_rmsnorm_zc(const char *op, const void *x, const float *w, void *out, int64_t rows, int64_t d, int groups,
                       float eps, Act act, hipStream_t stream) {
    dim3 grid((unsigned)rows), block(kThreads);
    switch (act) {
        case Act::F32:
            rmsnorm_zc_kernel<float><<<grid, block, 0, stream>>>(static_cast<const float *>(x), w,
                                                                 static_cast<float *>(out), d, groups, eps);
            break;
        case Act::BF16:
            rmsnorm_zc_kernel<uint16_t><<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(x), w,
                                                                    static_cast<uint16_t *>(out), d, groups, eps);
            break;
        default: STRIX_FAIL(op, ": unsupported activation dtype ", (int)act);
    }
}

template <typename T>
__global__ void __launch_bounds__(kThreads)
    rmsnorm_gated_kernel(const T *x, const T *z, const float *w, T *out, int64_t d, float eps, bool sigmoid_gate,
                         int64_t z_heads, int64_t z_ts) {
    __shared__ float red[kThreads];
    int64_t base = (int64_t)blockIdx.x * d;
    const int64_t zbase = z_heads > 0 ? (int64_t)blockIdx.x / z_heads * z_ts + (int64_t)blockIdx.x % z_heads * d : base;
    float inv = row_inv_rms(x + base, d, eps, red);
    for (int64_t j = threadIdx.x; j < d; j += blockDim.x) {
        float n = round_act<T>(load(x, base + j) * inv);  // reference casts the normed value back first
        float zv = load(z, zbase + j);
        // silu keeps its original form z / (1 + e^-z): z * sigmoid(z) rounds differently, and the Qwen3.5
        // BF16 forward is sensitive to that (a one-ulp shift here moved its logits error by ~28%).
        const float den = 1.0f + expf(-zv);  // sigmoid_gate is uniform across the block: no divergence
        store(out, base + j, (w[j] * n) * (sigmoid_gate ? 1.0f / den : zv / den));
    }
}

// The GDN gate norm's shape (BF16, d = 128): a wave per row, kGatedRows rows a block, lane l holding elements
// 4l .. 4l + 3 (one 8-byte load of x and of z, one 8-byte store). The block-per-row kernel above leaves half its 256
// threads idle on a 128-wide row.
// Bit-identical to it: the square sum follows that kernel's LDS tree order for d = 128 exactly (its threads 128..255
// add zeros; then s = 64, 32, .., 1), here as P[i] = sq[i] + sq[i + 64] (lane l + 16), P[i] + P[i + 32] (lane l + 8),
// then + 16, + 8, + 4 (lanes l + 4, + 2, + 1) and + 2, + 1 inside the lane. The squares use __fmul_rn so the compiler
// can't fuse a square into the next add (the tree adds already-rounded squares; a BF16 value's square is exact in
// FP32, so this only matters if the input type ever widens). Wave32 (gfx11) is checked on the host.
constexpr int kGatedD = 128, kGatedRows = 8;

__device__ inline float4 shfl_down4(float4 v, int s) {
    return make_float4(__shfl_down(v.x, s, 32), __shfl_down(v.y, s, 32), __shfl_down(v.z, s, 32),
                       __shfl_down(v.w, s, 32));
}
__device__ inline float4 add4(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }

__global__ void __launch_bounds__(kGatedRows * 32)
    rmsnorm_gated_d128_kernel(const uint16_t *x, const uint16_t *z, const float *w, uint16_t *out, int64_t rows,
                              float eps, bool sigmoid_gate, int64_t z_heads, int64_t z_ts) {
    const int lane = (int)threadIdx.x % 32;
    const int64_t row = (int64_t)blockIdx.x * kGatedRows + threadIdx.x / 32;
    if (row >= rows) return;  // wave-uniform
    const int64_t base = row * kGatedD + 4 * lane;
    const int64_t zbase = (z_heads > 0 ? row / z_heads * z_ts + row % z_heads * kGatedD : row * kGatedD) + 4 * lane;
    const uint2 xb = *reinterpret_cast<const uint2 *>(x + base), zb = *reinterpret_cast<const uint2 *>(z + zbase);
    const float xv[4] = {__uint_as_float(xb.x << 16), __uint_as_float(xb.x & 0xffff0000u),
                         __uint_as_float(xb.y << 16), __uint_as_float(xb.y & 0xffff0000u)};
    float4 p = make_float4(__fmul_rn(xv[0], xv[0]), __fmul_rn(xv[1], xv[1]), __fmul_rn(xv[2], xv[2]),
                           __fmul_rn(xv[3], xv[3]));
    p = add4(p, shfl_down4(p, 16));  // i + 64: valid in lanes 0..15
    p = add4(p, shfl_down4(p, 8));   // i + 32: lanes 0..7
    p = add4(p, shfl_down4(p, 4));   // i + 16: lanes 0..3
    p = add4(p, shfl_down4(p, 2));   // i + 8: lanes 0..1
    p = add4(p, shfl_down4(p, 1));   // i + 4: lane 0
    const float s2x = p.x + p.z, s2y = p.y + p.w;  // i + 2
    const float total = __shfl(s2x + s2y, 0, 32);  // i + 1, broadcast
    const float inv = 1.0f / sqrtf(total / (float)kGatedD + eps);
    const float4 wv = *reinterpret_cast<const float4 *>(w + 4 * lane);
    const float ww[4] = {wv.x, wv.y, wv.z, wv.w};
    const float zv[4] = {__uint_as_float(zb.x << 16), __uint_as_float(zb.x & 0xffff0000u),
                         __uint_as_float(zb.y << 16), __uint_as_float(zb.y & 0xffff0000u)};
    uint16_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {  // the same per-element expression as rmsnorm_gated_kernel
        const float n = round_act<uint16_t>(xv[i] * inv);
        const float den = 1.0f + expf(-zv[i]);
        o[i] = f32_to_bf16_bits((ww[i] * n) * (sigmoid_gate ? 1.0f / den : zv[i] / den));
    }
    *reinterpret_cast<uint2 *>(out + base) =
        make_uint2((uint32_t)o[0] | ((uint32_t)o[1] << 16), (uint32_t)o[2] | ((uint32_t)o[3] << 16));
}

// The d128 kernel's preconditions; anything else runs rmsnorm_gated_kernel.
bool gated_d128_fits(const void *x, const void *z, const float *w, const void *out, int64_t d, Act act, int64_t z_heads,
                     int64_t z_ts) {
    auto a = [](const void *p, uintptr_t n) { return reinterpret_cast<uintptr_t>(p) % n == 0; };
    return act == Act::BF16 && d == kGatedD && a(x, 8) && a(z, 8) && a(out, 8) && a(w, 16) &&
           (z_heads == 0 || z_ts % 4 == 0);
}

void check_common(const char *op, const void *x, const float *w, const void *out, int64_t rows, int64_t d, float eps) {
    STRIX_CHECK(x != nullptr && w != nullptr && out != nullptr, op, ": null pointer (x=", x, ", w=", w, ", out=", out,
                ")");
    STRIX_CHECK(rows >= 1 && rows <= kMaxRows, op, ": rows = ", rows, ", expected 1..", kMaxRows);
    STRIX_CHECK(d >= 1 && d <= kMaxRowLen, op, ": d = ", d, ", expected 1..", kMaxRowLen);
    STRIX_CHECK(std::isfinite(eps) && eps > 0.f, op, ": eps must be finite and > 0, got ", eps);
}

}  // namespace

const char *act_name(Act a) {
    switch (a) {
        case Act::F32: return "F32";
        case Act::BF16: return "BF16";
    }
    return "?";
}

void rmsnorm_zc(const void *x, const float *w, void *out, int64_t rows, int64_t d, float eps, Act act,
                hipStream_t stream) {
    check_common("rmsnorm_zc", x, w, out, rows, d, eps);
    launch_rmsnorm_zc("rmsnorm_zc", x, w, out, rows, d, 1, eps, act, stream);
    STRIX_HIP_CHECK_LAUNCH("rmsnorm_zc ", act_name(act), " rows=", rows, " d=", d);
}

void group_rmsnorm_zc(const void *x, const float *w, void *out, int64_t tokens, int64_t groups, int64_t d, float eps,
                      Act act, hipStream_t stream) {
    STRIX_CHECK(groups >= 1 && groups <= kMaxGroups, "group_rmsnorm_zc: groups = ", groups, ", expected 1..",
                kMaxGroups);
    STRIX_CHECK(tokens >= 1 && tokens <= kMaxRows / groups, "group_rmsnorm_zc: tokens = ", tokens, ", expected 1..",
                kMaxRows / groups, " (tokens * groups = rows must fit the grid)");
    // rows/d/eps/pointers: the same limits as rmsnorm_zc, on the [tokens * groups, d] view.
    check_common("group_rmsnorm_zc", x, w, out, tokens * groups, d, eps);
    launch_rmsnorm_zc("group_rmsnorm_zc", x, w, out, tokens * groups, d, (int)groups, eps, act, stream);
    STRIX_HIP_CHECK_LAUNCH("group_rmsnorm_zc ", act_name(act), " tokens=", tokens, " groups=", groups, " d=", d);
}

const char *gate_act_name(GateAct g) {
    switch (g) {
        case GateAct::SiLU: return "silu";
        case GateAct::Sigmoid: return "sigmoid";
    }
    return "?";
}

void rmsnorm_gated(const void *x, const void *z, const float *w, void *out, int64_t rows, int64_t d, float eps,
                   GateAct gate, Act act, hipStream_t stream, int64_t z_heads, int64_t z_token_stride) {
    check_common("rmsnorm_gated", x, w, out, rows, d, eps);
    STRIX_CHECK(z != nullptr, "rmsnorm_gated: gate z is null");
    STRIX_CHECK(z_heads >= 0 && (z_heads == 0 || (rows % z_heads == 0 && z_token_stride >= z_heads * d)),
                "rmsnorm_gated: z layout z_heads = ", z_heads, ", z_token_stride = ", z_token_stride, " for rows = ", rows,
                ", d = ", d, " (expected z_heads 0, or dividing rows with a token stride >= z_heads * d)");
    STRIX_CHECK(gate == GateAct::SiLU || gate == GateAct::Sigmoid, "rmsnorm_gated: unknown gate activation ",
                (int)gate);
    const bool sig = gate == GateAct::Sigmoid;
    if (gated_d128_fits(x, z, w, out, d, act, z_heads, z_token_stride)) {
        STRIX_CHECK(device_warp_size(kGatedRows * 32) == 32, "rmsnorm_gated: the d128 kernel needs wave32, the device "
                    "runs wave", device_warp_size(kGatedRows * 32));
        dim3 grid((unsigned)((rows + kGatedRows - 1) / kGatedRows)), block(kGatedRows * 32);
        rmsnorm_gated_d128_kernel<<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(x),
                                                              static_cast<const uint16_t *>(z), w,
                                                              static_cast<uint16_t *>(out), rows, eps, sig, z_heads,
                                                              z_token_stride);
        STRIX_HIP_CHECK_LAUNCH("rmsnorm_gated d128 ", gate_act_name(gate), " rows=", rows);
        return;
    }
    dim3 grid((unsigned)rows), block(kThreads);
    switch (act) {
        case Act::F32:
            rmsnorm_gated_kernel<float><<<grid, block, 0, stream>>>(
                static_cast<const float *>(x), static_cast<const float *>(z), w, static_cast<float *>(out), d, eps, sig,
                z_heads, z_token_stride);
            break;
        case Act::BF16:
            rmsnorm_gated_kernel<uint16_t><<<grid, block, 0, stream>>>(static_cast<const uint16_t *>(x),
                                                                       static_cast<const uint16_t *>(z), w,
                                                                       static_cast<uint16_t *>(out), d, eps, sig,
                                                                       z_heads, z_token_stride);
            break;
        default: STRIX_FAIL("rmsnorm_gated: unsupported activation dtype ", (int)act);
    }
    STRIX_HIP_CHECK_LAUNCH("rmsnorm_gated ", gate_act_name(gate), " ", act_name(act), " rows=", rows, " d=", d);
}

}  // namespace strix::kernels
