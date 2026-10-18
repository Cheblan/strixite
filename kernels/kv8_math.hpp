#pragma once

// Device-side 8-bit K/V arithmetic (kernels/kv8.hpp has the format), shared by kernels/kv8.hip (quantize) and the
// attention kernels' readers (kernels/attention.hip), so every reader dequantizes identically.

#include "kernels/kv8.hpp"
#include "kernels/rope_math.hpp"  // BF16 rounding

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels::kv8_math {

__device__ inline float h2f(uint32_t bits16) { return (float)__builtin_bit_cast(_Float16, (uint16_t)bits16); }
__device__ inline uint16_t f2h(float f) { return __builtin_bit_cast(uint16_t, (_Float16)f); }  // RNE

// The (scale, zero) word of a range [lo, hi] of finite values; ok = false if a parameter overflows fp16.
__device__ inline uint32_t params(float lo, float hi, bool &ok) {
    // A zero-width range, or one under fp16's smallest step, rounds to scale 0 (code() then stores 0).
    const uint16_t zb = f2h(lo), sb = f2h((hi - lo) / 255.f);
    ok = isfinite(h2f(zb)) && isfinite(h2f(sb));
    return (uint32_t)sb | ((uint32_t)zb << 16);
}
__device__ inline uint8_t code(float x, uint32_t p) {
    const float s = h2f(p & 0xffffu), z = h2f(p >> 16);
    if (s == 0.f) return 0;
    return (uint8_t)fminf(fmaxf(rintf((x - z) / s), 0.f), 255.f);
}
__device__ inline float value(uint32_t c, uint32_t p) { return fmaf((float)c, h2f(p & 0xffffu), h2f(p >> 16)); }

// 8 values of one row as 16 bytes in BF16 (T = uint16_t) or 4 values as FP32 (T = float), from 8-bit codes and
// one param word per value (K: per channel) or one for all of them (V: they lie in one block).
template <typename T>
__device__ inline uint4 pack(const float *f);
template <>
__device__ inline uint4 pack<uint16_t>(const float *f) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        w[i] = (uint32_t)rope_math::f32_to_bf16_bits(f[2 * i]) | ((uint32_t)rope_math::f32_to_bf16_bits(f[2 * i + 1]) << 16);
    return make_uint4(w[0], w[1], w[2], w[3]);
}
template <>
__device__ inline uint4 pack<float>(const float *f) {
    return make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]), __float_as_uint(f[2]), __float_as_uint(f[3]));
}

// The first row a query at position p sees exact (kernels/kv8.hpp): the start of the group p + 1 doesn't complete.
__host__ __device__ inline int64_t exact_from(int64_t p) { return (p + 1) / kKv8Group * kKv8Group; }

// Reader over a Kv8View for one layer: k(row, col, g0) / v(row, col, g0) return the 16 bytes (16 / sizeof(T)
// values) of row `row` from element col (a multiple of 16 / sizeof(T)) as the query with exact_from = g0 sees it:
// dequantized below g0, else exact - from the tail before the call (rows below pos0) or the call's new rows.
template <typename T>
struct Kv8Reader {
    static constexpr int kVec = 16 / sizeof(T);
    const uint8_t *kc;
    const uint32_t *kp;
    const uint8_t *vc;
    const uint32_t *vp;
    const T *tk, *tv, *nk, *nv;
    int64_t W, t0, pos0, new_ts;
    __host__ __device__ Kv8Reader(const Kv8View &x)
        : kc(x.store), kp(reinterpret_cast<const uint32_t *>(x.store + x.capacity * x.W)),
          vc(x.store + x.capacity * x.W + (x.capacity / kKv8Group) * x.W * 4),
          vp(reinterpret_cast<const uint32_t *>(vc + x.capacity * x.W)), tk(static_cast<const T *>(x.tail_in)),
          tv(static_cast<const T *>(x.tail_in) + kKv8Group * x.W), nk(static_cast<const T *>(x.k_new)),
          nv(static_cast<const T *>(x.v_new)), W(x.W), t0(x.pos0 / kKv8Group * kKv8Group), pos0(x.pos0),
          new_ts(x.new_ts) {}
    __device__ inline uint4 exact(const T *tail, const T *fresh, int64_t row, int64_t col) const {
        return row < pos0 ? *reinterpret_cast<const uint4 *>(tail + (row - t0) * W + col)
                          : *reinterpret_cast<const uint4 *>(fresh + (row - pos0) * new_ts + col);
    }
    __device__ inline void codes(const uint8_t *p, uint32_t (&c)[kVec]) const {
        if constexpr (kVec == 8) {
            const uint2 u = *reinterpret_cast<const uint2 *>(p);
#pragma unroll
            for (int i = 0; i < 4; ++i) c[i] = (u.x >> (8 * i)) & 0xffu, c[4 + i] = (u.y >> (8 * i)) & 0xffu;
        } else {
            const uint32_t u = *reinterpret_cast<const uint32_t *>(p);
#pragma unroll
            for (int i = 0; i < 4; ++i) c[i] = (u >> (8 * i)) & 0xffu;
        }
    }
    __device__ inline uint4 k(int64_t row, int64_t col, int64_t g0) const {
        if (row >= g0) return exact(tk, nk, row, col);
        uint32_t c[kVec];
        codes(kc + row * W + col, c);
        const uint32_t *p = kp + (row / kKv8Group) * W + col;
        float f[kVec];
#pragma unroll
        for (int i = 0; i < kVec; ++i) f[i] = value(c[i], p[i]);
        return pack<T>(f);
    }
    __device__ inline uint4 v(int64_t row, int64_t col, int64_t g0) const {
        if (row >= g0) return exact(tv, nv, row, col);
        uint32_t c[kVec];
        codes(vc + row * W + col, c);
        const uint32_t p = vp[row * (W / kKv8Block) + col / kKv8Block];
        float f[kVec];
#pragma unroll
        for (int i = 0; i < kVec; ++i) f[i] = value(c[i], p);
        return pack<T>(f);
    }
};

// Kv8Reader over a staged view (Kv8View::staged_k / staged_v): rows below g0 come from the dequantized copy - one
// 16-byte load, no unpacking - the rest exact as in Kv8Reader. Equal to Kv8Reader value for value.
template <typename T>
struct StagedReader {
    Kv8Reader<T> base;
    const T *sk, *sv;
    __host__ __device__ StagedReader(const Kv8View &x)
        : base(x), sk(static_cast<const T *>(x.staged_k)), sv(static_cast<const T *>(x.staged_v)) {}
    __device__ inline uint4 k(int64_t row, int64_t col, int64_t g0) const {
        return row >= g0 ? base.exact(base.tk, base.nk, row, col)
                         : *reinterpret_cast<const uint4 *>(sk + row * base.W + col);
    }
    __device__ inline uint4 v(int64_t row, int64_t col, int64_t g0) const {
        return row >= g0 ? base.exact(base.tv, base.nv, row, col)
                         : *reinterpret_cast<const uint4 *>(sv + row * base.W + col);
    }
};

// The BF16 / FP32 cache as the same interface (today's [capacity, Hk, D] layout: row r at r W).
template <typename T>
struct PlainReader {
    const T *kc, *vc;
    int64_t W;
    __device__ inline uint4 k(int64_t row, int64_t col, int64_t) const { return *reinterpret_cast<const uint4 *>(kc + row * W + col); }
    __device__ inline uint4 v(int64_t row, int64_t col, int64_t) const { return *reinterpret_cast<const uint4 *>(vc + row * W + col); }
};

}  // namespace strix::kernels::kv8_math
