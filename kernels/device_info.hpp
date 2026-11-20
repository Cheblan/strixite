#pragma once

// Host-side device facts the kernel launchers size grids from.

#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"

namespace strix::kernels {

// Wavefront size of the current device (32 on gfx11's default wave32), queried once. Checks it divides
// block_threads, the launcher's block size.
inline int device_warp_size(int block_threads) {
    static int cached = 0;
    if (!cached) {
        int dev = 0;
        STRIX_HIP_CHECK(hipGetDevice(&dev), "querying the current device");
        hipDeviceProp_t prop;
        STRIX_HIP_CHECK(hipGetDeviceProperties(&prop, dev), "querying device ", dev, " properties");
        STRIX_CHECK(prop.warpSize == 32 || prop.warpSize == 64, "unexpected wavefront size ", prop.warpSize);
        cached = prop.warpSize;
    }
    STRIX_CHECK(block_threads % cached == 0, "block size ", block_threads, " not a multiple of wavefront size ",
                cached);
    return cached;
}

// Multiprocessors as HIP reports them (WGPs on gfx1151: 20 for 40 CUs).
inline int device_mps() {
    static int cached = 0;
    if (!cached) {
        int dev = 0, v = 0;
        STRIX_HIP_CHECK(hipGetDevice(&dev), "current device");
        STRIX_HIP_CHECK(hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev), "MP count");
        STRIX_CHECK(v >= 1, "device reports ", v, " multiprocessors");
        cached = v;
    }
    return cached;
}

}  // namespace strix::kernels
