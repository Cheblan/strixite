#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q6.hpp"
#include "kernels/linear_q6.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>
#include <iomanip>

using namespace strix;
using namespace strix::kernels;

inline float bf16_bits_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

void test_q6_shape(int64_t M, int64_t N, int64_t K, int64_t G) {
    std::cout << "Testing linear_q6: M=" << M << ", N=" << N << ", K=" << K << ", G=" << G << " ... " << std::flush;

    // 1. Create pseudo-random Q6 weights on CPU
    Q6Weight w;
    w.N = N;
    w.K = K;
    w.G = G;
    int64_t row_b = q6_row_bytes(K);
    w.q.resize(N * row_b);
    for (size_t i = 0; i < w.q.size(); ++i) w.q[i] = (uint8_t)(i * 31 + 7);

    int64_t groups = N * (K / G);
    w.scale.resize(groups);
    w.minv.resize(groups);
    for (int64_t g = 0; g < groups; ++g) {
        w.scale[g] = f32_to_bf16_bits(0.015625f * (1.0f + (g % 8) * 0.125f)); // non-trivial scales
        w.minv[g]  = f32_to_bf16_bits(-0.5f + (g % 5) * 0.1f);
    }

    // 2. CPU Reference dequantization
    std::vector<float> w_dequant = dequantize_q6(w);

    // 3. Create input activations X
    std::vector<uint16_t> x_bf16(M * K);
    std::vector<float> x_f32(M * K);
    for (int64_t i = 0; i < M * K; ++i) {
        float val = ((i % 17) - 8.0f) * 0.25f;
        x_bf16[i] = f32_to_bf16_bits(val);
        x_f32[i] = bf16_bits_to_f32(x_bf16[i]); // exact BF16 value
    }

    // 4. CPU Reference MatMul: Y = X * W^T
    std::vector<float> y_ref(M * N, 0.0f);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += x_f32[m * K + k] * w_dequant[n * K + k];
            }
            y_ref[m * N + n] = sum;
        }
    }

    // 5. GPU Allocations
    uint8_t *d_q = nullptr;
    uint16_t *d_scale = nullptr, *d_minv = nullptr, *d_x = nullptr, *d_y = nullptr;
    STRIX_HIP_CHECK(hipMalloc(&d_q, w.q.size()));
    STRIX_HIP_CHECK(hipMalloc(&d_scale, w.scale.size() * sizeof(uint16_t)));
    STRIX_HIP_CHECK(hipMalloc(&d_minv, w.minv.size() * sizeof(uint16_t)));
    STRIX_HIP_CHECK(hipMalloc(&d_x, x_bf16.size() * sizeof(uint16_t)));
    STRIX_HIP_CHECK(hipMalloc(&d_y, M * N * sizeof(uint16_t)));

    STRIX_HIP_CHECK(hipMemcpy(d_q, w.q.data(), w.q.size(), hipMemcpyHostToDevice));
    STRIX_HIP_CHECK(hipMemcpy(d_scale, w.scale.data(), w.scale.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    STRIX_HIP_CHECK(hipMemcpy(d_minv, w.minv.data(), w.minv.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    STRIX_HIP_CHECK(hipMemcpy(d_x, x_bf16.data(), x_bf16.size() * sizeof(uint16_t), hipMemcpyHostToDevice));

    hipStream_t stream;
    STRIX_HIP_CHECK(hipStreamCreate(&stream));

    Q6DeviceView dv;
    dv.q = d_q;
    dv.scale = d_scale;
    dv.minv = d_minv;
    dv.N = N;
    dv.K = K;
    dv.G = G;

    // Run GPU kernel
    linear_q6(d_x, dv, d_y, M, Act::BF16, stream);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream));

    // Read back GPU output
    std::vector<uint16_t> y_gpu_bf16(M * N);
    STRIX_HIP_CHECK(hipMemcpy(y_gpu_bf16.data(), d_y, M * N * sizeof(uint16_t), hipMemcpyDeviceToHost));

    // Compare GPU vs CPU Reference
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;
    for (int64_t i = 0; i < M * N; ++i) {
        float yg = bf16_bits_to_f32(y_gpu_bf16[i]);
        float yr = y_ref[i];
        float diff = std::abs(yg - yr);
        float rel = (std::abs(yr) > 1e-4f) ? diff / std::abs(yr) : diff;
        if (diff > max_diff) max_diff = diff;
        if (rel > max_rel_diff) max_rel_diff = rel;
    }

    if (max_rel_diff > 0.005f) { // 0.5% tolerance due to dot2 summation order in BF16
        std::cout << "FAILED! Max Rel Diff = " << max_rel_diff << ", Max Diff = " << max_diff << "\n";
        std::exit(1);
    } else {
        std::cout << "PASSED (Max Rel Diff = " << std::scientific << std::setprecision(2) << max_rel_diff << ")\n";
    }

    STRIX_HIP_CHECK(hipStreamDestroy(stream));
    STRIX_HIP_CHECK(hipFree(d_q));
    STRIX_HIP_CHECK(hipFree(d_scale));
    STRIX_HIP_CHECK(hipFree(d_minv));
    STRIX_HIP_CHECK(hipFree(d_x));
    STRIX_HIP_CHECK(hipFree(d_y));
}

int main() {
    std::cout << "=================================================================\n";
    std::cout << "   TEST LINEAR_Q6: ACCURACY AND BOUNDS VERIFICATION ON GPU       \n";
    std::cout << "=================================================================\n";

    // Test across various M (MTP verify sizes and batched sizes)
    for (int m : {1, 2, 3, 4, 5, 8, 16}) {
        test_q6_shape(m, 256, 2560, 64);
    }

    // Test larger N
    test_q6_shape(1, 4096, 2560, 64);
    test_q6_shape(4, 4096, 2560, 64);

    std::cout << "\nALL LINEAR_Q6 ACCURACY TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
