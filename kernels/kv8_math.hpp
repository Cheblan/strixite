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

// f rounded to nearest even BF16, in the high 16 bits (the low 16 are junk). Only for a finite f: no Inf / NaN case,
// so no branch - rope_math::f32_to_bf16_bits's test made each value a divergent if / else. Every value() is finite (a
// code 0..255 times a finite fp16 scale plus a finite fp16 zero: |value| < 2^24), so this equals f32_to_bf16_bits.
__device__ inline uint32_t bf16_high_finite(float f) {
    const uint32_t u = __float_as_uint(f);
    return u + 0x7fffu + ((u >> 16) & 1u);
}

// 8 values of one row as 16 bytes in BF16 (T = uint16_t) or 4 values as FP32 (T = float), from 8-bit codes and
// one param word per value (K: per channel) or one for all of them (V: they lie in one block). f: value()s (finite).
template <typename T>
__device__ inline uint4 pack(const float *f);
template <>
__device__ inline uint4 pack<uint16_t>(const float *f) {
    uint32_t w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        w[i] = (bf16_high_finite(f[2 * i]) >> 16) | (bf16_high_finite(f[2 * i + 1]) & 0xffff0000u);
    return make_uint4(w[0], w[1], w[2], w[3]);
}
template <>
__device__ inline uint4 pack<float>(const float *f) {
    return make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]), __float_as_uint(f[2]), __float_as_uint(f[3]));
}

// The WMMA attention's K over the 8-bit cache is F16, not BF16 (its scores run on v_wmma_f32_16x16x16_f16, q
// converted to F16 too): a dequantized value rounds to F16 in 3 VALU ops a pair (2 v_cvt_f16_f32 + a pack) where the
// BF16 RNE took ~5 a value with no BF16 convert instruction on gfx11 - that unpacking was ~90% of the 8-bit cost over
// BF16 at verify sizes. F16 keeps 11 bits to BF16's 8, so the K values are closer to the dequantized FP32 ones. The
// caller guarantees |K| and |q| fit F16's normal range (the model checks its q / k norm weights at load:
// Qwen4ExpModel, kF16AttnBound).
// a, b rounded to nearest even F16, packed (a low).
__device__ inline uint32_t f16x2(float a, float b) { return (uint32_t)f2h(a) | ((uint32_t)f2h(b) << 16); }
// 8 BF16 values (16 bytes) as 8 F16 values - exact for values in F16's normal range.
__device__ inline uint4 bf16_to_f16x8(uint4 x) {
    const uint32_t w[4] = {x.x, x.y, x.z, x.w};
    uint32_t o[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = f16x2(__uint_as_float(w[i] << 16), __uint_as_float(w[i] & 0xffff0000u));
    return make_uint4(o[0], o[1], o[2], o[3]);
}

// The first row a query at position p sees exact (kernels/kv8.hpp): the start of the group p + 1 doesn't complete.
__host__ __device__ inline int64_t exact_from(int64_t p) { return (p + 1) / kKv8Group * kKv8Group; }

// Reader over a Kv8View for one layer: k(row, col, g0) / v(row, col, g0) return the 16 bytes (16 / sizeof(T)
// values) of row `row` from element col (a multiple of 16 / sizeof(T)) as the query with exact_from = g0 sees it:
// dequantized below g0, else exact - from the tail before the call (rows below pos0) or the call's new rows.
template <typename T>
struct Kv8Reader {
    static constexpr int kVec = 16 / sizeof(T);
    static constexpr bool kStaged = false;  // no plain-load range (StagedReader has one)
    static constexpr bool kKF16 = sizeof(T) == 2;  // k16 (the WMMA attention's F16 K) - BF16 activations only
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
    // The 8-bit rows by 32-bit offsets - every one is below capacity * W < 2^32 (check_kv) - so a load costs no
    // 64-bit multiply or signed divide.
    __device__ inline uint4 k(int64_t row, int64_t col, int64_t g0) const {
        if (row >= g0) return exact(tk, nk, row, col);
        const uint32_t r = (uint32_t)row, cl = (uint32_t)col, w = (uint32_t)W;
        uint32_t c[kVec];
        codes(kc + (r * w + cl), c);
        const uint32_t *p = kp + (r / (uint32_t)kKv8Group * w + cl);
        float f[kVec];
#pragma unroll
        for (int i = 0; i < kVec; ++i) f[i] = value(c[i], p[i]);
        return pack<T>(f);
    }
    // k()'s 8 values (T = uint16_t) as F16: dequantized rows rounded once from FP32 to F16, exact rows converted.
    __device__ inline uint4 k16(int64_t row, int64_t col, int64_t g0) const {
        static_assert(kVec == 8, "Kv8Reader::k16: BF16 activations only");
        if (row >= g0) return bf16_to_f16x8(exact(tk, nk, row, col));
        return k16_from(kcodes(row, col), kparams(row, col));
    }
    // The pieces of k16 for a reader that loads a row's codes and its params apart (the WMMA attention: codes a tile
    // ahead in registers, params through LDS) - k16_from(kcodes, kparams) is k16 bit for bit.
    __device__ inline uint2 kcodes(int64_t row, int64_t col) const {
        return *reinterpret_cast<const uint2 *>(kc + ((uint32_t)row * (uint32_t)W + (uint32_t)col));
    }
    __device__ inline const uint32_t *kparams(int64_t row, int64_t col) const {
        return kp + ((uint32_t)row / (uint32_t)kKv8Group * (uint32_t)W + (uint32_t)col);
    }
    __device__ inline uint4 kexact(int64_t row, int64_t col) const { return exact(tk, nk, row, col); }
    // 8 codes (uint2, channel col .. col + 7 of one row) and their 8 per-channel param words as F16 values.
    __device__ static inline uint4 k16_from(uint2 u, const uint32_t *p) {
        uint32_t o[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t w = i < 2 ? u.x : u.y, sh = 16 * (i % 2);
            o[i] = f16x2(value((w >> sh) & 0xffu, p[2 * i]), value((w >> (sh + 8)) & 0xffu, p[2 * i + 1]));
        }
        return make_uint4(o[0], o[1], o[2], o[3]);
    }
    __device__ inline uint4 v(int64_t row, int64_t col, int64_t g0) const {
        if (row >= g0) return exact(tv, nv, row, col);
        const uint32_t r = (uint32_t)row, cl = (uint32_t)col, w = (uint32_t)W;
        uint32_t c[kVec];
        codes(vc + (r * w + cl), c);
        const uint32_t p = vp[r * (w / (uint32_t)kKv8Block) + cl / (uint32_t)kKv8Block];
        float f[kVec];
#pragma unroll
        for (int i = 0; i < kVec; ++i) f[i] = value(c[i], p);
        return pack<T>(f);
    }
};

// Kv8Reader over a staged view (Kv8View::staged_k / staged_v, the WMMA attention's prefill): rows below g0 come from
// the dequantized copy - one 16-byte load, no unpacking - the rest exact as in Kv8Reader. The copy's K is F16
// (kv8_dequant k_f16), so K is read through k16 only; equal to Kv8Reader's k16 / v value for value. BF16 only.
template <typename T>
struct StagedReader {
    static_assert(sizeof(T) == 2, "StagedReader: the staged copy is F16 K / BF16 V");
    Kv8Reader<T> base;
    const T *sk, *sv;
    __host__ __device__ StagedReader(const Kv8View &x)
        : base(x), sk(static_cast<const T *>(x.staged_k)), sv(static_cast<const T *>(x.staged_v)) {}
    // ks / vs: a row known to lie below g0 (a caller that tested it once for many loads), the staged copy with no test.
    static constexpr bool kStaged = true;
    static constexpr bool kKF16 = true;
    __device__ inline uint4 ks(int64_t row, int64_t col) const { return *reinterpret_cast<const uint4 *>(sk + row * base.W + col); }
    __device__ inline uint4 vs(int64_t row, int64_t col) const { return *reinterpret_cast<const uint4 *>(sv + row * base.W + col); }
    // The same by a 32-bit byte offset from the copy's base (row * W + col) * sizeof(T) - the caller's offsets fit 32
    // bits (check_kv: a staged copy is under 4 GiB), so a load is a uniform base + one 32-bit VGPR (+ an immediate)
    // with no 64-bit address arithmetic.
    __device__ inline uint32_t row_bytes() const { return (uint32_t)(base.W * (int64_t)sizeof(T)); }
    __device__ inline uint4 ks_at(uint32_t byte) const { return *reinterpret_cast<const uint4 *>(reinterpret_cast<const char *>(sk) + byte); }
    __device__ inline uint4 vs_at(uint32_t byte) const { return *reinterpret_cast<const uint4 *>(reinterpret_cast<const char *>(sv) + byte); }
    __device__ inline uint4 k16(int64_t row, int64_t col, int64_t g0) const {
        return row >= g0 ? bf16_to_f16x8(base.exact(base.tk, base.nk, row, col)) : ks(row, col);
    }
    __device__ inline uint4 v(int64_t row, int64_t col, int64_t g0) const {
        return row >= g0 ? base.exact(base.tv, base.nv, row, col) : vs(row, col);
    }
};

// The BF16 / FP32 cache as the same interface (today's [capacity, Hk, D] layout: row r at r W).
template <typename T>
struct PlainReader {
    static constexpr bool kStaged = false;
    static constexpr bool kKF16 = false;
    const T *kc, *vc;
    int64_t W;
    __device__ inline uint4 k(int64_t row, int64_t col, int64_t) const { return *reinterpret_cast<const uint4 *>(kc + row * W + col); }
    __device__ inline uint4 v(int64_t row, int64_t col, int64_t) const { return *reinterpret_cast<const uint4 *>(vc + row * W + col); }
};

}  // namespace strix::kernels::kv8_math
