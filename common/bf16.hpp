#pragma once

// Host-side BF16 bit helpers: the memcpy twins of the device functions of the same name in
// kernels/rope_math.hpp, with the identical semantics (round-to-nearest-even, non-finite values preserved -
// the same form the kernels' own rounding uses). For host code that converts the BF16 values the kernels
// store: the loader's widenings, the vision pixel upload, host-side reference computations.

#include <cstdint>
#include <cstring>

namespace strix {

inline float bf16_bits_to_f32(uint16_t bits) {
    const uint32_t u = (uint32_t)bits << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_bf16_bits(float v) {
    uint32_t u;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

}  // namespace strix
