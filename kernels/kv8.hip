#include "kernels/kv8.hpp"

#include "common/hip_check.hpp"
#include "kernels/kv8_math.hpp"

#include <cstdint>

namespace strix::kernels {

namespace {

using rope_math::bf16_bits_to_f32;
using rope_math::f32_to_bf16_bits;

constexpr int64_t kMaxW = 1024;  // one thread per channel of a group: one workgroup

__device__ inline float ld(const float *p, int64_t i) { return p[i]; }
__device__ inline float ld(const uint16_t *p, int64_t i) { return bf16_bits_to_f32(p[i]); }

__device__ inline void flag(uint32_t *err, int64_t row, int64_t ch) {
    if (atomicCAS(err, 0u, 1u) == 0u) err[1] = (uint32_t)row, err[2] = (uint32_t)ch;
}

// One workgroup per group completed by the call (group index gb + blockIdx.x), W threads. Thread c: K channel c over
// the group's 32 rows, and V block c % (W / 32) of the group's row c / (W / 32). Row r of the group comes from
// tail_in (r < pos0: the rows before this call, tail_in row r - g0(pos0)) or from the new rows.
template <typename T>
__global__ void kv8_groups_kernel(const T *k_new, const T *v_new, int64_t new_ts, int64_t pos0, const T *tail_in,
                                  int64_t gb, uint8_t *kc, uint32_t *kp, uint8_t *vc, uint32_t *vp, int64_t W,
                                  uint32_t *err) {
    const int64_t c = threadIdx.x, g = gb + blockIdx.x, r0 = g * kKv8Group, t0 = (pos0 / kKv8Group) * kKv8Group;
    auto src = [&](bool is_k, int64_t r, int64_t ch) -> float {
        if (r < pos0) return ld(tail_in, (is_k ? 0 : kKv8Group * W) + (r - t0) * W + ch);
        return ld(is_k ? k_new : v_new, (r - pos0) * new_ts + ch);
    };
    // K: channel c.
    {
        float x[kKv8Group], lo, hi;
#pragma unroll
        for (int i = 0; i < kKv8Group; ++i) x[i] = src(true, r0 + i, c);
        lo = hi = x[0];
#pragma unroll
        for (int i = 1; i < kKv8Group; ++i) lo = fminf(lo, x[i]), hi = fmaxf(hi, x[i]);
        bool ok = true;
        const uint32_t p = kv8_math::params(lo, hi, ok);
        if (!ok) flag(err, r0, c);
        kp[g * W + c] = p;
#pragma unroll
        for (int i = 0; i < kKv8Group; ++i) kc[(r0 + i) * W + c] = kv8_math::code(x[i], p);
    }
    // V: one block of one row.
    {
        const int64_t nb = W / kKv8Block, r = r0 + c / nb, b0 = (c % nb) * kKv8Block;
        float x[kKv8Block], lo, hi;
#pragma unroll
        for (int i = 0; i < kKv8Block; ++i) x[i] = src(false, r, b0 + i);
        lo = hi = x[0];
#pragma unroll
        for (int i = 1; i < kKv8Block; ++i) lo = fminf(lo, x[i]), hi = fmaxf(hi, x[i]);
        bool ok = true;
        const uint32_t p = kv8_math::params(lo, hi, ok);
        if (!ok) flag(err, r, b0);
        vp[r * nb + c % nb] = p;
#pragma unroll
        for (int i = 0; i < kKv8Block; ++i) vc[r * W + b0 + i] = kv8_math::code(x[i], p);
    }
}

// tail_out rows [g1, pos0 + T) (K then V halves), from tail_in or the new rows.
template <typename T>
__global__ void kv8_tail_kernel(const T *k_new, const T *v_new, int64_t new_ts, int64_t pos0, const T *tail_in,
                                T *tail_out, int64_t g1, int64_t n, int64_t W) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * n * W) return;
    const int64_t half = i / (n * W), rr = (i / W) % n, ch = i % W, r = g1 + rr;
    const int64_t at = half * kKv8Group * W + rr * W + ch, t0 = (pos0 / kKv8Group) * kKv8Group;
    tail_out[at] = r < pos0 ? tail_in[half * kKv8Group * W + (r - t0) * W + ch]
                            : (half ? v_new : k_new)[(r - pos0) * new_ts + ch];
}

template <typename T>
__global__ void kv8_dequant_kernel(const uint8_t *store, int64_t capacity, int64_t W, int64_t r0, int64_t n, T *dk,
                                   T *dv) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * W) return;
    const int64_t r = r0 + i / W, ch = i % W;
    const uint8_t *kc = store;
    const uint32_t *kp = reinterpret_cast<const uint32_t *>(store + capacity * W);
    const uint8_t *vc = store + capacity * W + (capacity / kKv8Group) * W * 4;
    const uint32_t *vp = reinterpret_cast<const uint32_t *>(vc + capacity * W);
    const float k = kv8_math::value(kc[r * W + ch], kp[(r / kKv8Group) * W + ch]);
    const float v = kv8_math::value(vc[r * W + ch], vp[r * (W / kKv8Block) + ch / kKv8Block]);
    if constexpr (sizeof(T) == 2) dk[i] = f32_to_bf16_bits(k), dv[i] = f32_to_bf16_bits(v);
    else dk[i] = k, dv[i] = v;
}

size_t es_of(const char *fn, Act act) {
    STRIX_CHECK(act == Act::BF16 || act == Act::F32, fn, ": activation dtype ", (int)act, ", expected BF16 or F32");
    return act == Act::F32 ? 4 : 2;
}

void check_layout(const char *fn, int64_t capacity, int64_t W) {
    STRIX_CHECK(W >= kKv8Block && W <= kMaxW && W % kKv8Block == 0, fn, ": W = ", W, ", expected a multiple of ",
                kKv8Block, " up to ", kMaxW);
    STRIX_CHECK(capacity >= kKv8Group && capacity % kKv8Group == 0, fn, ": capacity ", capacity,
                ", expected a positive multiple of ", kKv8Group);
}

bool overlaps(const void *a, size_t an, const void *b, size_t bn) {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x < y + bn && y < x + an;
}

}  // namespace

size_t kv8_bytes(int64_t capacity, int64_t W) {
    check_layout("kv8_bytes", capacity, W);
    return (size_t)(capacity * W) * 2 + (size_t)((capacity / kKv8Group) * W) * 4 +
           (size_t)(capacity * (W / kKv8Block)) * 4;
}

size_t kv8_tail_bytes(int64_t W, Act act) {
    check_layout("kv8_tail_bytes", kKv8Group, W);
    return (size_t)(2 * kKv8Group * W) * es_of("kv8_tail_bytes", act);
}

void kv8_commit(const void *k_new, const void *v_new, int64_t new_ts, int64_t T, int64_t pos0, const void *tail_in,
                void *tail_out, uint8_t *store, int64_t capacity, int64_t W, Act act, uint32_t *err,
                hipStream_t stream) {
    const char *fn = "kv8_commit";
    const size_t es = es_of(fn, act);
    check_layout(fn, capacity, W);
    STRIX_CHECK(k_new && v_new && tail_out && store && err && (tail_in || pos0 % kKv8Group == 0), fn,
                ": null pointer (k_new=", k_new, ", v_new=", v_new, ", tail_in=", tail_in, ", tail_out=", tail_out,
                ", store=", (const void *)store, ", err=", (const void *)err, ")");
    STRIX_CHECK(T >= 1 && pos0 >= 0 && pos0 + T <= capacity, fn, ": rows [", pos0, ", ", pos0 + T,
                ") outside the capacity ", capacity);
    STRIX_CHECK(new_ts >= W, fn, ": new-row stride ", new_ts, " < W = ", W);
    STRIX_CHECK(reinterpret_cast<uintptr_t>(store) % 16 == 0 && reinterpret_cast<uintptr_t>(tail_out) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(tail_in) % 16 == 0,
                fn, ": store / tails must be 16-byte aligned (store=", (const void *)store, ", tail_in=", tail_in,
                ", tail_out=", tail_out, ")");
    const size_t tb = kv8_tail_bytes(W, act);
    STRIX_CHECK(tail_in == nullptr || !overlaps(tail_in, tb, tail_out, tb), fn,
                ": tail_out overlaps tail_in (two buffers: a dropped verify keeps tail_in)");
    STRIX_CHECK(!overlaps(tail_out, tb, k_new, (size_t)((T - 1) * new_ts + W) * es) &&
                    !overlaps(tail_out, tb, v_new, (size_t)((T - 1) * new_ts + W) * es),
                fn, ": tail_out overlaps the new rows");
    const int64_t ga = pos0 / kKv8Group, gb = (pos0 + T) / kKv8Group, g1 = gb * kKv8Group, n = pos0 + T - g1;
    uint8_t *kc = store, *vc = store + capacity * W + (capacity / kKv8Group) * W * 4;
    uint32_t *kp = reinterpret_cast<uint32_t *>(store + capacity * W), *vp = reinterpret_cast<uint32_t *>(vc + capacity * W);
    if (act == Act::BF16) {
        const auto *kn = static_cast<const uint16_t *>(k_new), *vn = static_cast<const uint16_t *>(v_new),
                   *ti = static_cast<const uint16_t *>(tail_in);
        if (gb > ga)
            hipLaunchKernelGGL(kv8_groups_kernel<uint16_t>, dim3((unsigned)(gb - ga)), dim3((unsigned)W), 0, stream, kn,
                               vn, new_ts, pos0, ti, ga, kc, kp, vc, vp, W, err);
        if (n > 0)
            hipLaunchKernelGGL(kv8_tail_kernel<uint16_t>, dim3((unsigned)((2 * n * W + 255) / 256)), dim3(256), 0,
                               stream, kn, vn, new_ts, pos0, ti, static_cast<uint16_t *>(tail_out), g1, n, W);
    } else {
        const auto *kn = static_cast<const float *>(k_new), *vn = static_cast<const float *>(v_new),
                   *ti = static_cast<const float *>(tail_in);
        if (gb > ga)
            hipLaunchKernelGGL(kv8_groups_kernel<float>, dim3((unsigned)(gb - ga)), dim3((unsigned)W), 0, stream, kn,
                               vn, new_ts, pos0, ti, ga, kc, kp, vc, vp, W, err);
        if (n > 0)
            hipLaunchKernelGGL(kv8_tail_kernel<float>, dim3((unsigned)((2 * n * W + 255) / 256)), dim3(256), 0, stream,
                               kn, vn, new_ts, pos0, ti, static_cast<float *>(tail_out), g1, n, W);
    }
    STRIX_HIP_CHECK(hipGetLastError(), fn, ": launch (T = ", T, ", pos0 = ", pos0, ", W = ", W, ")");
}

void report_kv8_error(const uint32_t e[3], const char *what) {
    STRIX_CHECK(e[0] == 0, what, ": 8-bit K/V quantization parameter outside fp16 at row ", e[1], ", channel ", e[2],
                " (a K/V value range beyond fp16 - the stored codes of that group are meaningless)");
}

void kv8_dequant(const uint8_t *store, int64_t capacity, int64_t W, int64_t r0, int64_t r1, void *dst_k, void *dst_v,
                 Act act, hipStream_t stream) {
    const char *fn = "kv8_dequant";
    es_of(fn, act);
    check_layout(fn, capacity, W);
    STRIX_CHECK(store && dst_k && dst_v, fn, ": null pointer (store=", (const void *)store, ", dst_k=", dst_k,
                ", dst_v=", dst_v, ")");
    STRIX_CHECK(r0 >= 0 && r0 <= r1 && r1 <= capacity, fn, ": rows [", r0, ", ", r1, ") outside [0, ", capacity, ")");
    const int64_t n = r1 - r0;
    if (n == 0) return;
    const dim3 grid((unsigned)((n * W + 255) / 256));
    if (act == Act::BF16)
        hipLaunchKernelGGL(kv8_dequant_kernel<uint16_t>, grid, dim3(256), 0, stream, store, capacity, W, r0, n,
                           static_cast<uint16_t *>(dst_k), static_cast<uint16_t *>(dst_v));
    else
        hipLaunchKernelGGL(kv8_dequant_kernel<float>, grid, dim3(256), 0, stream, store, capacity, W, r0, n,
                           static_cast<float *>(dst_k), static_cast<float *>(dst_v));
    STRIX_HIP_CHECK(hipGetLastError(), fn, ": launch (rows [", r0, ", ", r1, "), W = ", W, ")");
}

}  // namespace strix::kernels
