// Test optimizations for hc_inject_inv_kernel on gfx1151.
// Verifies 100% bit-identity against baseline and measures latency.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "kernels/device_info.hpp"
#include "kernels/wmma_gemm.hpp"
#include "runtime/device_buffer.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

constexpr int kH = 4;
constexpr int kInvThreads = 256;

__device__ inline float to_f32(float v) { return v; }
__device__ inline float to_f32(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }
__device__ inline void store(float *p, int64_t i, float v) { p[i] = v; }
__device__ inline void store(uint16_t *p, int64_t i, float v) { p[i] = wmma_f32_to_bf16(v); }

// Baseline kernel from kernels/hc_wmma.hip
template <typename T>
__global__ void __launch_bounds__(kInvThreads)
    hc_inject_inv_baseline_kernel(T *x, const float *w_in, const T *y, int64_t d, float eps, float *inv) {
    __shared__ float red[kInvThreads];
    const int64_t t = blockIdx.x;
    for (int s = 0; s < kH; ++s) {
        T *xs = x + (t * kH + s) * d;
        const float w = w_in[t * kH + s];
        float ss = 0.f;
        for (int64_t j = threadIdx.x; j < d; j += kInvThreads) {
            store(xs, j, fmaf(w, to_f32(y[t * d + j]), to_f32(xs[j])));
            const float v = to_f32(xs[j]);
            ss = fmaf(v, v, ss);
        }
        red[threadIdx.x] = ss;
        __syncthreads();
        for (int off = kInvThreads / 2; off > 0; off >>= 1) {
            if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) inv[t * kH + s] = 1.0f / sqrtf(red[0] / (float)d + eps);
        __syncthreads();
    }
}

// Variant 1: Cached y in registers + register v (no global reload)
template <typename T>
__global__ void __launch_bounds__(kInvThreads)
    hc_inject_inv_opt1_kernel(T *x, const float *w_in, const T *y, int64_t d, float eps, float *inv) {
    __shared__ float red[kInvThreads];
    const int64_t t = blockIdx.x;
    
    // Preload y into registers once per token (10 elements for d = 2560, kInvThreads = 256)
    constexpr int kIters = 10;
    T y_reg[kIters];
    #pragma unroll
    for (int i = 0; i < kIters; ++i) {
        int64_t j = threadIdx.x + (int64_t)i * kInvThreads;
        y_reg[i] = (j < d) ? y[t * d + j] : (T)0;
    }

    for (int s = 0; s < kH; ++s) {
        T *xs = x + (t * kH + s) * d;
        const float w = w_in[t * kH + s];
        float ss = 0.f;
        #pragma unroll
        for (int i = 0; i < kIters; ++i) {
            int64_t j = threadIdx.x + (int64_t)i * kInvThreads;
            if (j < d) {
                float res = fmaf(w, to_f32(y_reg[i]), to_f32(xs[j]));
                T rounded = (T)wmma_f32_to_bf16(res);
                xs[j] = rounded;
                float v = to_f32(rounded);
                ss = fmaf(v, v, ss);
            }
        }
        red[threadIdx.x] = ss;
        __syncthreads();
        for (int off = kInvThreads / 2; off > 0; off >>= 1) {
            if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) inv[t * kH + s] = 1.0f / sqrtf(red[0] / (float)d + eps);
        __syncthreads();
    }
}

// Variant 2: Cached y + register v + Wave-shuffle reduction (exact bit identity)
template <typename T>
__global__ void __launch_bounds__(kInvThreads)
    hc_inject_inv_opt2_kernel(T *x, const float *w_in, const T *y, int64_t d, float eps, float *inv) {
    __shared__ float red[kInvThreads];
    const int64_t t = blockIdx.x;
    
    constexpr int kIters = 10;
    T y_reg[kIters];
    #pragma unroll
    for (int i = 0; i < kIters; ++i) {
        int64_t j = threadIdx.x + (int64_t)i * kInvThreads;
        y_reg[i] = (j < d) ? y[t * d + j] : (T)0;
    }

    for (int s = 0; s < kH; ++s) {
        T *xs = x + (t * kH + s) * d;
        const float w = w_in[t * kH + s];
        float ss = 0.f;
        #pragma unroll
        for (int i = 0; i < kIters; ++i) {
            int64_t j = threadIdx.x + (int64_t)i * kInvThreads;
            if (j < d) {
                float res = fmaf(w, to_f32(y_reg[i]), to_f32(xs[j]));
                T rounded = (T)wmma_f32_to_bf16(res);
                xs[j] = rounded;
                float v = to_f32(rounded);
                ss = fmaf(v, v, ss);
            }
        }
        red[threadIdx.x] = ss;
        __syncthreads();
        
        // Tree reduction down to 32 threads (Wave 0)
        if (threadIdx.x < 128) red[threadIdx.x] += red[threadIdx.x + 128];
        __syncthreads();
        if (threadIdx.x < 64) red[threadIdx.x] += red[threadIdx.x + 64];
        __syncthreads();
        if (threadIdx.x < 32) red[threadIdx.x] += red[threadIdx.x + 32];
        __syncthreads();
        
        // Wave 0 shuffle reduction (exact bit-identical tree to off = 16, 8, 4, 2, 1)
        if (threadIdx.x < 32) {
            float val = red[threadIdx.x];
            val += __shfl_down(val, 16, 32);
            val += __shfl_down(val, 8, 32);
            val += __shfl_down(val, 4, 32);
            val += __shfl_down(val, 2, 32);
            val += __shfl_down(val, 1, 32);
            if (threadIdx.x == 0) inv[t * kH + s] = 1.0f / sqrtf(val / (float)d + eps);
        }
        __syncthreads();
    }
}

// Variant 3: Vectorized 128-bit (uint4 = 8 BF16 elements) loads/stores
__global__ void __launch_bounds__(kInvThreads)
    hc_inject_inv_vec128_kernel(uint16_t *x, const float *w_in, const uint16_t *y, int64_t d, float eps, float *inv) {
    __shared__ float red[kInvThreads];
    const int64_t t = blockIdx.x;
    
    // Each thread processes (d / 8) / 256 chunks = 320 / 256 chunks
    // Thread tid processes chunk tid, and if tid < 64 also chunk tid + 256
    const int64_t num_chunks = d / 8; // 320
    const uint4 *y_vec = reinterpret_cast<const uint4 *>(y + t * d);
    
    for (int s = 0; s < kH; ++s) {
        uint4 *xs_vec = reinterpret_cast<uint4 *>(x + (t * kH + s) * d);
        const float w = w_in[t * kH + s];
        float ss = 0.f;
        
        for (int64_t c = threadIdx.x; c < num_chunks; c += kInvThreads) {
            uint4 y_val = y_vec[c];
            uint4 x_val = xs_vec[c];
            const uint16_t *yb = reinterpret_cast<const uint16_t *>(&y_val);
            const uint16_t *xb = reinterpret_cast<const uint16_t *>(&x_val);
            uint16_t rb[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float res = fmaf(w, to_f32(yb[k]), to_f32(xb[k]));
                uint16_t rounded = wmma_f32_to_bf16(res);
                rb[k] = rounded;
                float v = to_f32(rounded);
                ss = fmaf(v, v, ss);
            }
            xs_vec[c] = *reinterpret_cast<const uint4 *>(rb);
        }
        
        red[threadIdx.x] = ss;
        __syncthreads();
        if (threadIdx.x < 128) red[threadIdx.x] += red[threadIdx.x + 128];
        __syncthreads();
        if (threadIdx.x < 64) red[threadIdx.x] += red[threadIdx.x + 64];
        __syncthreads();
        if (threadIdx.x < 32) red[threadIdx.x] += red[threadIdx.x + 32];
        __syncthreads();
        if (threadIdx.x < 32) {
            float val = red[threadIdx.x];
            val += __shfl_down(val, 16, 32);
            val += __shfl_down(val, 8, 32);
            val += __shfl_down(val, 4, 32);
            val += __shfl_down(val, 2, 32);
            val += __shfl_down(val, 1, 32);
            if (threadIdx.x == 0) inv[t * kH + s] = 1.0f / sqrtf(val / (float)d + eps);
        }
        __syncthreads();
    }
}

uint16_t host_float_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u)
        return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    const int64_t T = 8192;
    const int64_t H = 4;
    const int64_t d = 2560;
    const float eps = 1e-6f;

    std::printf("================================================================================\n");
    std::printf("   HC_INJECT_INV OPTIMIZATION AND BIT-IDENTITY TEST ON GFX1151                  \n");
    std::printf("   T = %ld tokens, H = %ld streams, d = %ld (Activation size = 167.8 MB)         \n", (long)T, (long)H, (long)d);
    std::printf("================================================================================\n\n");

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> host_x_init((size_t)(T * H * d));
    for (size_t i = 0; i < host_x_init.size(); ++i) host_x_init[i] = host_float_to_bf16(dist(rng));

    std::vector<uint16_t> host_y((size_t)(T * d));
    for (size_t i = 0; i < host_y.size(); ++i) host_y[i] = host_float_to_bf16(dist(rng));

    std::vector<float> host_w((size_t)(T * H));
    for (size_t i = 0; i < host_w.size(); ++i) host_w[i] = dist(rng);

    DeviceBuffer<uint16_t> d_x_base = DeviceBuffer<uint16_t>::from_host(host_x_init, "x_base");
    DeviceBuffer<uint16_t> d_x_opt1 = DeviceBuffer<uint16_t>::from_host(host_x_init, "x_opt1");
    DeviceBuffer<uint16_t> d_x_opt2 = DeviceBuffer<uint16_t>::from_host(host_x_init, "x_opt2");
    DeviceBuffer<uint16_t> d_x_vec  = DeviceBuffer<uint16_t>::from_host(host_x_init, "x_vec");

    DeviceBuffer<uint16_t> d_y = DeviceBuffer<uint16_t>::from_host(host_y, "y");
    DeviceBuffer<float> d_w = DeviceBuffer<float>::from_host(host_w, "w_in");

    DeviceBuffer<float> d_inv_base((size_t)(T * H), "inv_base");
    DeviceBuffer<float> d_inv_opt1((size_t)(T * H), "inv_opt1");
    DeviceBuffer<float> d_inv_opt2((size_t)(T * H), "inv_opt2");
    DeviceBuffer<float> d_inv_vec ((size_t)(T * H), "inv_vec");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    // 1. Run baseline
    hc_inject_inv_baseline_kernel<uint16_t><<<(unsigned)T, kInvThreads, 0, stream>>>(
        d_x_base.get(), d_w.get(), d_y.get(), d, eps, d_inv_base.get());
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "baseline run");

    // 2. Run Opt1 (cached y)
    hc_inject_inv_opt1_kernel<uint16_t><<<(unsigned)T, kInvThreads, 0, stream>>>(
        d_x_opt1.get(), d_w.get(), d_y.get(), d, eps, d_inv_opt1.get());
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "opt1 run");

    // 3. Run Opt2 (cached y + wave shuffle)
    hc_inject_inv_opt2_kernel<uint16_t><<<(unsigned)T, kInvThreads, 0, stream>>>(
        d_x_opt2.get(), d_w.get(), d_y.get(), d, eps, d_inv_opt2.get());
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "opt2 run");

    // 4. Run Vec128
    hc_inject_inv_vec128_kernel<<<(unsigned)T, kInvThreads, 0, stream>>>(
        d_x_vec.get(), d_w.get(), d_y.get(), d, eps, d_inv_vec.get());
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "vec run");

    // Verification against baseline
    std::vector<uint16_t> h_x_base((size_t)(T * H * d)), h_x_opt1((size_t)(T * H * d)),
                          h_x_opt2((size_t)(T * H * d)), h_x_vec((size_t)(T * H * d));
    std::vector<float> h_inv_base((size_t)(T * H)), h_inv_opt1((size_t)(T * H)),
                       h_inv_opt2((size_t)(T * H)), h_inv_vec((size_t)(T * H));

    STRIX_HIP_CHECK(hipMemcpy(h_x_base.data(), d_x_base.get(), h_x_base.size() * 2, hipMemcpyDeviceToHost), "cpy");
    STRIX_HIP_CHECK(hipMemcpy(h_x_opt1.data(), d_x_opt1.get(), h_x_opt1.size() * 2, hipMemcpyDeviceToHost), "cpy");
    STRIX_HIP_CHECK(hipMemcpy(h_x_opt2.data(), d_x_opt2.get(), h_x_opt2.size() * 2, hipMemcpyDeviceToHost), "cpy");
    STRIX_HIP_CHECK(hipMemcpy(h_x_vec.data(),  d_x_vec.get(),  h_x_vec.size() * 2, hipMemcpyDeviceToHost), "cpy");

    STRIX_HIP_CHECK(hipMemcpy(h_inv_base.data(), d_inv_base.get(), h_inv_base.size() * 4, hipMemcpyDeviceToHost), "cpy");
    STRIX_HIP_CHECK(hipMemcpy(h_inv_opt1.data(), d_inv_opt1.get(), h_inv_opt1.size() * 4, hipMemcpyDeviceToHost), "cpy");
    STRIX_HIP_CHECK(hipMemcpy(h_inv_opt2.data(), d_inv_opt2.get(), h_inv_opt2.size() * 4, hipMemcpyDeviceToHost), "cpy");
    STRIX_HIP_CHECK(hipMemcpy(h_inv_vec.data(),  d_inv_vec.get(),  h_inv_vec.size() * 4, hipMemcpyDeviceToHost), "cpy");

    auto check_diffs = [&](const char *label, const std::vector<uint16_t> &x_test, const std::vector<float> &inv_test) {
        int64_t x_diffs = 0;
        for (size_t i = 0; i < x_test.size(); ++i) {
            if (x_test[i] != h_x_base[i]) ++x_diffs;
        }
        int64_t inv_diffs = 0;
        float max_inv_diff = 0.0f;
        for (size_t i = 0; i < inv_test.size(); ++i) {
            uint32_t b_base, b_test;
            std::memcpy(&b_base, &h_inv_base[i], 4);
            std::memcpy(&b_test, &inv_test[i], 4);
            if (b_base != b_test) {
                ++inv_diffs;
                float d = std::abs(h_inv_base[i] - inv_test[i]);
                if (d > max_inv_diff) max_inv_diff = d;
            }
        }
        std::printf("Verification: %-25s -> X diffs: %ld / %zu (%.4f%%), inv diffs: %ld / %zu, max_inv_diff: %g\n",
                    label, (long)x_diffs, x_test.size(), (double)x_diffs / x_test.size() * 100.0,
                    (long)inv_diffs, inv_test.size(), (double)max_inv_diff);
    };

    check_diffs("Opt1 (cached y)", h_x_opt1, h_inv_opt1);
    check_diffs("Opt2 (cached y + shfl)", h_x_opt2, h_inv_opt2);
    check_diffs("Vec128 (128-bit chunks)", h_x_vec, h_inv_vec);

    // Benchmarking
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
        std::printf("Benchmark:    %-25s -> %7.3f ms (96 calls = %6.2f ms total)\n",
                    label, avg_ms, avg_ms * 96.0f);
        return avg_ms;
    };

    std::printf("\n--- Performance Timings (avg over 30 iterations) ---\n");
    bench([&]() {
        hc_inject_inv_baseline_kernel<uint16_t><<<(unsigned)T, kInvThreads, 0, stream>>>(
            d_x_base.get(), d_w.get(), d_y.get(), d, eps, d_inv_base.get());
    }, "Baseline");

    bench([&]() {
        hc_inject_inv_opt1_kernel<uint16_t><<<(unsigned)T, kInvThreads, 0, stream>>>(
            d_x_opt1.get(), d_w.get(), d_y.get(), d, eps, d_inv_opt1.get());
    }, "Opt1 (cached y)");

    bench([&]() {
        hc_inject_inv_opt2_kernel<uint16_t><<<(unsigned)T, kInvThreads, 0, stream>>>(
            d_x_opt2.get(), d_w.get(), d_y.get(), d, eps, d_inv_opt2.get());
    }, "Opt2 (cached y + shfl)");

    bench([&]() {
        hc_inject_inv_vec128_kernel<<<(unsigned)T, kInvThreads, 0, stream>>>(
            d_x_vec.get(), d_w.get(), d_y.get(), d, eps, d_inv_vec.get());
    }, "Vec128 (128-bit chunks)");

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return 0;
}
