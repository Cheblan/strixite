// Standalone verification and benchmark tool for Fused GDN Decode Step.
// Compares the baseline 3-kernel pipeline (gdn_front + delta_rule + rmsnorm_gated)
// against the streamlined pipeline with delta_rule_rmsnorm_fused for decode (T=1).

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "kernels/gdn.hpp"
#include "kernels/norm.hpp"

#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

using namespace strix;
using namespace strix::kernels;

namespace {

inline float bf16_to_f32(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    ::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    ::memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1;
    u += 0x7fff + lsb;
    return static_cast<uint16_t>(u >> 16);
}

} // namespace

int main() {
    std::printf("================================================================================\n");
    std::printf("              FUSED GDN DECODE STEP (T=1) BENCHMARK & VERIFICATION              \n");
    std::printf("================================================================================\n");

    int device_id = 0;
    STRIX_HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t prop;
    STRIX_HIP_CHECK(hipGetDeviceProperties(&prop, device_id));
    std::printf("Running on device: %s (%s)\n", prop.name, prop.gcnArchName);

    // Qwen4Exp GDN Dimensions
    const int64_t T = 1;
    const int64_t Hk = 16;
    const int64_t Hv = 48;
    const int64_t D = 128;
    const int64_t conv_c = 2 * Hk * D + Hv * D; // 10240
    const int64_t gz = Hv * D;                  // 6144
    const int64_t gstride = conv_c + gz + 2 * Hv; // 16480
    const int64_t b_col = conv_c + gz;          // 16384
    const int64_t a_col = b_col + Hv;           // 16432
    const float eps = 1e-6f;

    hipStream_t stream = nullptr;
    STRIX_HIP_CHECK(hipStreamCreate(&stream));

    // Host buffers
    std::vector<uint16_t> h_in_proj(gstride);
    std::vector<float> h_conv_w(conv_c * 4);
    std::vector<uint16_t> h_conv_state_init(3 * conv_c);
    std::vector<float> h_A_log(Hv);
    std::vector<float> h_dt_bias(Hv);
    std::vector<float> h_rec_state_init(Hv * D * D);
    std::vector<float> h_gdn_norm(D);

    uint32_t seed = 42;
    auto rnd = [&]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return ((float)(seed & 0xffff) / 65536.0f - 0.5f) * 0.2f;
    };

    for (size_t i = 0; i < h_in_proj.size(); ++i) h_in_proj[i] = f32_to_bf16(rnd());
    for (size_t i = 0; i < h_conv_w.size(); ++i) h_conv_w[i] = rnd() + 0.25f;
    for (size_t i = 0; i < h_conv_state_init.size(); ++i) h_conv_state_init[i] = f32_to_bf16(rnd());
    for (size_t i = 0; i < h_A_log.size(); ++i) h_A_log[i] = -0.5f + rnd() * 0.1f;
    for (size_t i = 0; i < h_dt_bias.size(); ++i) h_dt_bias[i] = 0.1f + rnd() * 0.1f;
    for (size_t i = 0; i < h_rec_state_init.size(); ++i) h_rec_state_init[i] = rnd() * 0.1f;
    for (size_t i = 0; i < h_gdn_norm.size(); ++i) h_gdn_norm[i] = 1.0f + rnd() * 0.05f;

    // Device allocations
    uint16_t *d_in_proj = nullptr;
    float *d_conv_w = nullptr;
    uint16_t *d_conv_state = nullptr;
    float *d_A_log = nullptr;
    float *d_dt_bias = nullptr;
    float *d_rec_state_ref = nullptr, *d_rec_state_fused = nullptr;
    float *d_gdn_norm = nullptr;

    uint16_t *d_qkv = nullptr;
    float *d_beta = nullptr;
    float *d_g = nullptr;
    uint16_t *d_core = nullptr;
    uint16_t *d_gnorm_ref = nullptr, *d_gnorm_fused = nullptr;

    STRIX_HIP_CHECK(hipMalloc(&d_in_proj, gstride * 2));
    STRIX_HIP_CHECK(hipMalloc(&d_conv_w, conv_c * 4 * 4));
    STRIX_HIP_CHECK(hipMalloc(&d_conv_state, 3 * conv_c * 2));
    STRIX_HIP_CHECK(hipMalloc(&d_A_log, Hv * 4));
    STRIX_HIP_CHECK(hipMalloc(&d_dt_bias, Hv * 4));
    STRIX_HIP_CHECK(hipMalloc(&d_rec_state_ref, Hv * D * D * 4));
    STRIX_HIP_CHECK(hipMalloc(&d_rec_state_fused, Hv * D * D * 4));
    STRIX_HIP_CHECK(hipMalloc(&d_gdn_norm, D * 4));

    STRIX_HIP_CHECK(hipMalloc(&d_qkv, conv_c * 2));
    STRIX_HIP_CHECK(hipMalloc(&d_beta, Hv * 4));
    STRIX_HIP_CHECK(hipMalloc(&d_g, Hv * 4));
    STRIX_HIP_CHECK(hipMalloc(&d_core, gz * 2));
    STRIX_HIP_CHECK(hipMalloc(&d_gnorm_ref, gz * 2));
    STRIX_HIP_CHECK(hipMalloc(&d_gnorm_fused, gz * 2));

    // Initialize device memory
    STRIX_HIP_CHECK(hipMemcpy(d_in_proj, h_in_proj.data(), gstride * 2, hipMemcpyHostToDevice));
    STRIX_HIP_CHECK(hipMemcpy(d_conv_w, h_conv_w.data(), conv_c * 4 * 4, hipMemcpyHostToDevice));
    STRIX_HIP_CHECK(hipMemcpy(d_conv_state, h_conv_state_init.data(), 3 * conv_c * 2, hipMemcpyHostToDevice));
    STRIX_HIP_CHECK(hipMemcpy(d_A_log, h_A_log.data(), Hv * 4, hipMemcpyHostToDevice));
    STRIX_HIP_CHECK(hipMemcpy(d_dt_bias, h_dt_bias.data(), Hv * 4, hipMemcpyHostToDevice));
    STRIX_HIP_CHECK(hipMemcpy(d_rec_state_ref, h_rec_state_init.data(), Hv * D * D * 4, hipMemcpyHostToDevice));
    STRIX_HIP_CHECK(hipMemcpy(d_rec_state_fused, h_rec_state_init.data(), Hv * D * D * 4, hipMemcpyHostToDevice));
    STRIX_HIP_CHECK(hipMemcpy(d_gdn_norm, h_gdn_norm.data(), D * 4, hipMemcpyHostToDevice));

    // 1. Run Baseline 3-kernel pipeline
    GdnFrontShape front_shape{T, Hk, D, Hv, D, gstride, b_col, a_col};
    gdn_front(d_in_proj, front_shape, d_conv_w, d_conv_state, d_conv_state, false, d_A_log, d_dt_bias,
              d_qkv, d_beta, d_g, Act::BF16, stream);

    DeltaRuleShape dr_shape{T, Hk, Hv, conv_c};
    delta_rule(d_qkv, d_beta, d_g, dr_shape, d_rec_state_ref, d_rec_state_ref, false, d_core, Act::BF16, stream,
               DeltaRulePath::Auto, DrPrecision::HalfU);

    rmsnorm_gated(d_core, d_in_proj + conv_c, d_gdn_norm, d_gnorm_ref, T * Hv, D, eps,
                  GateAct::Sigmoid, Act::BF16, stream, Hv, gstride);

    // 2. Run Fused GDN Decode: delta_rule_rmsnorm_fused
    delta_rule_rmsnorm_fused(d_qkv, d_beta, d_g, d_rec_state_fused, d_rec_state_fused,
                            d_in_proj + conv_c, d_gdn_norm, eps, d_gnorm_fused,
                            Hk, Hv, conv_c, gz, gstride, Act::BF16, stream);

    STRIX_HIP_CHECK(hipStreamSynchronize(stream));

    // 3. Verification & Numerical Comparison
    std::vector<float> h_rec_ref(Hv * D * D), h_rec_fused(Hv * D * D);
    std::vector<uint16_t> h_gnorm_ref_vec(gz), h_gnorm_fused_vec(gz);

    STRIX_HIP_CHECK(hipMemcpy(h_rec_ref.data(), d_rec_state_ref, Hv * D * D * 4, hipMemcpyDeviceToHost));
    STRIX_HIP_CHECK(hipMemcpy(h_rec_fused.data(), d_rec_state_fused, Hv * D * D * 4, hipMemcpyDeviceToHost));
    STRIX_HIP_CHECK(hipMemcpy(h_gnorm_ref_vec.data(), d_gnorm_ref, gz * 2, hipMemcpyDeviceToHost));
    STRIX_HIP_CHECK(hipMemcpy(h_gnorm_fused_vec.data(), d_gnorm_fused, gz * 2, hipMemcpyDeviceToHost));

    double max_rec_err = 0.0;
    int rec_diff = 0;
    for (size_t i = 0; i < h_rec_ref.size(); ++i) {
        double d = std::abs((double)h_rec_ref[i] - (double)h_rec_fused[i]);
        max_rec_err = std::max(max_rec_err, d);
        uint32_t a, b;
        ::memcpy(&a, &h_rec_ref[i], 4);
        ::memcpy(&b, &h_rec_fused[i], 4);
        if (a != b) ++rec_diff;
    }

    double max_gnorm_err = 0.0;
    int gnorm_diff = 0;
    for (size_t i = 0; i < h_gnorm_ref_vec.size(); ++i) {
        float r = bf16_to_f32(h_gnorm_ref_vec[i]);
        float f = bf16_to_f32(h_gnorm_fused_vec[i]);
        max_gnorm_err = std::max(max_gnorm_err, (double)std::abs(r - f));
        if (h_gnorm_ref_vec[i] != h_gnorm_fused_vec[i]) ++gnorm_diff;
    }

    std::printf("\nVERIFICATION RESULTS:\n");
    std::printf("  1. Rec State    : %s (%d / %zu elements differ, Max Abs Err: %10.4e)\n",
                rec_diff == 0 ? "100.00% BIT-EXACT MATCH" : "CLOSE NUMERICAL MATCH", rec_diff, h_rec_ref.size(), max_rec_err);
    std::printf("  2. GNorm Output : %s (%d / %zu elements differ, Max Abs Err: %10.4e)\n",
                gnorm_diff == 0 ? "100.00% BIT-EXACT MATCH" : "CLOSE NUMERICAL MATCH", gnorm_diff, h_gnorm_ref_vec.size(), max_gnorm_err);

    // 4. Benchmarking
    const int iters = 1000;
    hipEvent_t ev0, ev1;
    STRIX_HIP_CHECK(hipEventCreate(&ev0));
    STRIX_HIP_CHECK(hipEventCreate(&ev1));

    // gdn_front alone
    STRIX_HIP_CHECK(hipEventRecord(ev0, stream));
    for (int i = 0; i < iters; ++i) {
        gdn_front(d_in_proj, front_shape, d_conv_w, d_conv_state, d_conv_state, false, d_A_log, d_dt_bias,
                  d_qkv, d_beta, d_g, Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipEventRecord(ev1, stream));
    STRIX_HIP_CHECK(hipEventSynchronize(ev1));
    float ms_front = 0.0f;
    STRIX_HIP_CHECK(hipEventElapsedTime(&ms_front, ev0, ev1));
    float us_front = (ms_front / (float)iters) * 1000.0f;

    // delta_rule alone
    STRIX_HIP_CHECK(hipEventRecord(ev0, stream));
    for (int i = 0; i < iters; ++i) {
        delta_rule(d_qkv, d_beta, d_g, dr_shape, d_rec_state_ref, d_rec_state_ref, false, d_core, Act::BF16, stream,
                   DeltaRulePath::Auto, DrPrecision::HalfU);
    }
    STRIX_HIP_CHECK(hipEventRecord(ev1, stream));
    STRIX_HIP_CHECK(hipEventSynchronize(ev1));
    float ms_dr = 0.0f;
    STRIX_HIP_CHECK(hipEventElapsedTime(&ms_dr, ev0, ev1));
    float us_dr = (ms_dr / (float)iters) * 1000.0f;

    // rmsnorm_gated alone
    STRIX_HIP_CHECK(hipEventRecord(ev0, stream));
    for (int i = 0; i < iters; ++i) {
        rmsnorm_gated(d_core, d_in_proj + conv_c, d_gdn_norm, d_gnorm_ref, T * Hv, D, eps,
                      GateAct::Sigmoid, Act::BF16, stream, Hv, gstride);
    }
    STRIX_HIP_CHECK(hipEventRecord(ev1, stream));
    STRIX_HIP_CHECK(hipEventSynchronize(ev1));
    float ms_norm = 0.0f;
    STRIX_HIP_CHECK(hipEventElapsedTime(&ms_norm, ev0, ev1));
    float us_norm = (ms_norm / (float)iters) * 1000.0f;

    // Full baseline pipeline (gdn_front + delta_rule + rmsnorm_gated)
    STRIX_HIP_CHECK(hipEventRecord(ev0, stream));
    for (int i = 0; i < iters; ++i) {
        gdn_front(d_in_proj, front_shape, d_conv_w, d_conv_state, d_conv_state, false, d_A_log, d_dt_bias,
                  d_qkv, d_beta, d_g, Act::BF16, stream);
        delta_rule(d_qkv, d_beta, d_g, dr_shape, d_rec_state_ref, d_rec_state_ref, false, d_core, Act::BF16, stream,
                   DeltaRulePath::Auto, DrPrecision::HalfU);
        rmsnorm_gated(d_core, d_in_proj + conv_c, d_gdn_norm, d_gnorm_ref, T * Hv, D, eps,
                      GateAct::Sigmoid, Act::BF16, stream, Hv, gstride);
    }
    STRIX_HIP_CHECK(hipEventRecord(ev1, stream));
    STRIX_HIP_CHECK(hipEventSynchronize(ev1));
    float ms_pipe_base = 0.0f;
    STRIX_HIP_CHECK(hipEventElapsedTime(&ms_pipe_base, ev0, ev1));
    float us_pipe_base = (ms_pipe_base / (float)iters) * 1000.0f;

    // Fused pipeline (gdn_front + delta_rule_rmsnorm_fused)
    STRIX_HIP_CHECK(hipEventRecord(ev0, stream));
    for (int i = 0; i < iters; ++i) {
        gdn_front(d_in_proj, front_shape, d_conv_w, d_conv_state, d_conv_state, false, d_A_log, d_dt_bias,
                  d_qkv, d_beta, d_g, Act::BF16, stream);
        delta_rule_rmsnorm_fused(d_qkv, d_beta, d_g, d_rec_state_fused, d_rec_state_fused,
                                d_in_proj + conv_c, d_gdn_norm, eps, d_gnorm_fused,
                                Hk, Hv, conv_c, gz, gstride, Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipEventRecord(ev1, stream));
    STRIX_HIP_CHECK(hipEventSynchronize(ev1));
    float ms_pipe_fused = 0.0f;
    STRIX_HIP_CHECK(hipEventElapsedTime(&ms_pipe_fused, ev0, ev1));
    float us_pipe_fused = (ms_pipe_fused / (float)iters) * 1000.0f;

    std::printf("\nPERFORMANCE BREAKDOWN (Release -O3, 1000 iters):\n");
    std::printf("  1. gdn_front alone            : %6.2f us per layer (%5.2f ms / 32 layers)\n", us_front, (us_front * 32.0f) / 1000.0f);
    std::printf("  2. delta_rule alone           : %6.2f us per layer (%5.2f ms / 32 layers)\n", us_dr, (us_dr * 32.0f) / 1000.0f);
    std::printf("  3. rmsnorm_gated alone        : %6.2f us per layer (%5.2f ms / 32 layers)\n", us_norm, (us_norm * 32.0f) / 1000.0f);
    std::printf("  ------------------------------------------------------------------------\n");
    std::printf("  Baseline 3-kernel pipeline    : %6.2f us per layer (%5.2f ms / 32 layers)\n", us_pipe_base, (us_pipe_base * 32.0f) / 1000.0f);
    std::printf("  Fused 2-kernel pipeline       : %6.2f us per layer (%5.2f ms / 32 layers)\n", us_pipe_fused, (us_pipe_fused * 32.0f) / 1000.0f);
    std::printf("  Difference                    : %6.2f us per layer (%5.2f ms / 32 layers)\n", us_pipe_base - us_pipe_fused, ((us_pipe_base - us_pipe_fused) * 32.0f) / 1000.0f);
    std::printf("================================================================================\n");

    STRIX_HIP_CHECK(hipFree(d_in_proj));
    STRIX_HIP_CHECK(hipFree(d_conv_w));
    STRIX_HIP_CHECK(hipFree(d_conv_state));
    STRIX_HIP_CHECK(hipFree(d_A_log));
    STRIX_HIP_CHECK(hipFree(d_dt_bias));
    STRIX_HIP_CHECK(hipFree(d_rec_state_ref));
    STRIX_HIP_CHECK(hipFree(d_rec_state_fused));
    STRIX_HIP_CHECK(hipFree(d_gdn_norm));
    STRIX_HIP_CHECK(hipFree(d_qkv));
    STRIX_HIP_CHECK(hipFree(d_beta));
    STRIX_HIP_CHECK(hipFree(d_g));
    STRIX_HIP_CHECK(hipFree(d_core));
    STRIX_HIP_CHECK(hipFree(d_gnorm_ref));
    STRIX_HIP_CHECK(hipFree(d_gnorm_fused));
    STRIX_HIP_CHECK(hipStreamDestroy(stream));

    return (max_rec_err < 1e-6 && max_gnorm_err < 1e-3) ? 0 : 1;
}
