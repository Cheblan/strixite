// Comprehensive Layer Breakdown Benchmark on gfx1151 (Strix Halo).
// Measures every kernel in GDN, Attention, MoE, and Hyper-Connections at T = 8192.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "formats/q4.hpp"
#include "formats/q8.hpp"
#include "kernels/attention.hpp"
#include "kernels/gdn.hpp"
#include "kernels/hc_wmma.hpp"
#include "kernels/linear.hpp"
#include "kernels/linear_wmma.hpp"
#include "kernels/moe_grouped.hpp"
#include "kernels/moe_router.hpp"
#include "kernels/norm.hpp"
#include "kernels/swiglu.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"
#include "runtime/q8_device.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

Q4Device make_random_q4(int64_t N, int64_t K, int64_t G, std::mt19937 &rng, const std::string &name) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w((size_t)(N * K));
    for (float &v : w) v = dist(rng);
    Q4Weight qw = quantize_q4(w.data(), N, K, G);
    return Q4Device::upload(qw, name);
}

Q8Device make_random_q8(int64_t N, int64_t K, int64_t G, std::mt19937 &rng, const std::string &name) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w((size_t)(N * K));
    for (float &v : w) v = dist(rng);
    Q8Weight qw = quantize_q8(w.data(), N, K, G);
    return Q8Device::upload(qw, name);
}

Q8ChunkMajorDevice upload_random_q8_cm(int64_t N, int64_t K, int64_t G, std::mt19937 &rng, const std::string &name) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> w((size_t)(N * K));
    for (float &v : w) v = dist(rng);
    Q8Weight qw = quantize_q8(w.data(), N, K, G);
    Q8ChunkMajor qcm = to_chunk_major(qw, name.c_str());
    return Q8ChunkMajorDevice::upload(qcm, name);
}

void run_breakdown(int64_t T) {
    std::mt19937 rng(42);
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    auto time_kernel = [&](auto fn, const char *label, int iters = 15) -> float {
        // Warmup
        for (int i = 0; i < 2; ++i) fn();
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
        std::printf("  %-36s : %7.3f ms\n", label, avg_ms);
        return avg_ms;
    };

    std::printf("================================================================================\n");
    std::printf("   LAYER COMPONENT BREAKDOWN BENCHMARK AT T = %ld TOKENS (GFX1151)              \n", (long)T);
    std::printf("================================================================================\n\n");

    const int64_t d = 2560;
    const int64_t H = 4;
    const int64_t n4 = H * d; // 10240

    // --------------------------------------------------------------------------
    // 1. GDN Layer (36 layers)
    // --------------------------------------------------------------------------
    std::printf("--- 1. GDN Layer Breakdown (36 layers) ---\n");
    const int64_t gk = 16, gv = 48, kHD = 128;
    const int64_t conv_c = 10240, gz = 6144, gstride = 16480;

    // in_proj: Q4 [16480, 2560, G=128]
    Q4Device w_gdn_in = make_random_q4(gstride, d, 128, rng, "gdn_in");
    DeviceBuffer<uint16_t> x_u((size_t)(T * d), "u");
    DeviceBuffer<uint16_t> gdn_gp((size_t)(T * gstride), "gp");

    float ms_gdn_in = time_kernel([&]() {
        linear_q4_wmma(x_u.get(), w_gdn_in.view(), gdn_gp.get(), T, Act::BF16, Act::BF16, stream);
    }, "gdn_in (linear_q4_wmma [16480x2560])");

    // gdn_front
    std::vector<float> host_conv_w(conv_c * 4, 0.01f);
    std::vector<float> host_A_log(gv, -0.5f);
    std::vector<float> host_dt_bias(gv, 0.1f);
    DeviceBuffer<float> conv_w = DeviceBuffer<float>::from_host(host_conv_w, "conv_w");
    DeviceBuffer<float> A_log = DeviceBuffer<float>::from_host(host_A_log, "A_log");
    DeviceBuffer<float> dt_bias = DeviceBuffer<float>::from_host(host_dt_bias, "dt_bias");
    DeviceBuffer<uint16_t> conv_state((size_t)(3 * conv_c), "conv_state");
    DeviceBuffer<uint16_t> qkv_buf((size_t)(T * conv_c), "qkv_buf");
    DeviceBuffer<float> beta_buf((size_t)(T * gv), "beta");
    DeviceBuffer<float> g_buf((size_t)(T * gv), "g");

    GdnFrontShape gfs{T, gk, kHD, gv, kHD, gstride, conv_c + gz, conv_c + gz + gv};
    float ms_gdn_front = time_kernel([&]() {
        gdn_front(gdn_gp.get(), gfs, conv_w.get(), conv_state.get(), conv_state.get(), false,
                  A_log.get(), dt_bias.get(), qkv_buf.get(), beta_buf.get(), g_buf.get(), Act::BF16, stream);
    }, "gdn_front (conv causal + gates)");

    // delta_rule (ChunkedWmma, HalfU)
    DeviceBuffer<float> rec_state((size_t)(gv * kHD * kHD), "rec_state");
    DeviceBuffer<uint16_t> core_buf((size_t)(T * gz), "core");
    DeltaRuleShape drs{T, gk, gv, conv_c};

    float ms_delta_rule = time_kernel([&]() {
        delta_rule(qkv_buf.get(), beta_buf.get(), g_buf.get(), drs, rec_state.get(), rec_state.get(),
                   false, core_buf.get(), Act::BF16, stream, DeltaRulePath::ChunkedWmma, DrPrecision::HalfU);
    }, "delta_rule (ChunkedWmma / HalfU)");

    // rmsnorm_gated
    std::vector<float> host_gdn_norm(kHD, 1.0f);
    DeviceBuffer<float> gdn_norm = DeviceBuffer<float>::from_host(host_gdn_norm, "gdn_norm");
    DeviceBuffer<uint16_t> gnorm_buf((size_t)(T * gz), "gnorm");

    float ms_rmsnorm_gated = time_kernel([&]() {
        rmsnorm_gated(core_buf.get(), gdn_gp.get() + conv_c, gdn_norm.get(), gnorm_buf.get(),
                      T * gv, kHD, 1e-6f, GateAct::Sigmoid, Act::BF16, stream, gv, gstride);
    }, "rmsnorm_gated (RMSNorm + Sigmoid)");

    // gdn_out: Q8 [2560, 6144, G=64]
    Q8Device w_gdn_out = make_random_q8(d, gz, 64, rng, "gdn_out");
    DeviceBuffer<uint16_t> y_buf((size_t)(T * d), "y");

    float ms_gdn_out = time_kernel([&]() {
        linear_q8_wmma(gnorm_buf.get(), w_gdn_out.view(), y_buf.get(), T, Act::BF16, Act::BF16, stream);
    }, "gdn_out (linear_q8_wmma [2560x6144])");

    float ms_gdn_total = ms_gdn_in + ms_gdn_front + ms_delta_rule + ms_rmsnorm_gated + ms_gdn_out;
    std::printf("  TOTAL GDN Layer (single layer)       : %7.3f ms\n", ms_gdn_total);
    std::printf("  TOTAL GDN Layers (36 layers)          : %7.3f s per chunk\n\n", (ms_gdn_total * 36) / 1000.0f);

    // --------------------------------------------------------------------------
    // 2. Attention Layer (12 layers)
    // --------------------------------------------------------------------------
    std::printf("--- 2. Attention Layer Breakdown (12 layers) ---\n");
    const int64_t hq = 24, hkv = 2, hd = 256;
    const int64_t astride = 13952; // [13952 x 2560]
    Q8Device w_attn_qkv = make_random_q8(astride, d, 64, rng, "attn_qkv");
    DeviceBuffer<uint16_t> attn_proj((size_t)(T * astride), "attn_proj");

    float ms_attn_qkv = time_kernel([&]() {
        linear_q8_wmma(x_u.get(), w_attn_qkv.view(), attn_proj.get(), T, Act::BF16, Act::BF16, stream);
    }, "attn_qkv (linear_q8_wmma [13952x2560])");

    // qk_norm_rope_jobs
    std::vector<float> host_q_norm(hd, 1.0f), host_k_norm(hd, 1.0f);
    DeviceBuffer<float> q_norm = DeviceBuffer<float>::from_host(host_q_norm, "q_norm");
    DeviceBuffer<float> k_norm = DeviceBuffer<float>::from_host(host_k_norm, "k_norm");
    std::vector<float> host_inv_freq(hd / 2, 0.01f);
    DeviceBuffer<float> inv_freq = DeviceBuffer<float>::from_host(host_inv_freq, "inv_freq");
    DeviceBuffer<uint16_t> k_cache((size_t)(T * hkv * hd), "k_cache");
    DeviceBuffer<uint16_t> v_cache((size_t)(T * hkv * hd), "v_cache");

    NormRopeJob nr[3] = {
        {attn_proj.get(), astride, 2 * hd, attn_proj.get(), astride, 2 * hd, hq, hd, q_norm.get()},
        {attn_proj.get() + hq * 2 * hd, astride, hd, k_cache.get(), hkv * hd, hd, hkv, hd, k_norm.get()},
        {attn_proj.get() + hq * 2 * hd + hkv * hd, astride, hd, v_cache.get(), hkv * hd, hd, hkv, hd, nullptr}
    };

    float ms_qk_norm_rope = time_kernel([&]() {
        qk_norm_rope_jobs(nr, 3, T, 1e-6f, 64, 0, inv_freq.get(), Act::BF16, stream);
    }, "qk_norm_rope_jobs (q, k, v RoPE)");

    // attention_wmma
    AttentionShape as{T, 0, hq, hkv, hd, astride, 2 * hd};
    const float attn_scale = 1.0f / std::sqrt((float)hd);
    size_t attn_ws_bytes = (size_t)T * hq * hd * sizeof(float);
    DeviceBuffer<float> attn_ws(attn_ws_bytes / 4, "attn_ws");
    DeviceBuffer<uint16_t> attn_core((size_t)(T * hq * hd), "attn_core");

    float ms_attn_core = time_kernel([&]() {
        attention_wmma(attn_proj.get(), attn_proj.get() + hd, k_cache.get(), v_cache.get(),
                       T, as, attn_scale, attn_core.get(), attn_ws.get(), attn_ws_bytes, Act::BF16, stream);
    }, "attention_wmma (Softmax Attention)");

    // attn_o: Q8 [2560, 6144, G=64]
    Q8Device w_attn_o = make_random_q8(d, hq * hd, 64, rng, "attn_o");
    float ms_attn_o = time_kernel([&]() {
        linear_q8_wmma(attn_core.get(), w_attn_o.view(), y_buf.get(), T, Act::BF16, Act::BF16, stream);
    }, "attn_o (linear_q8_wmma [2560x6144])");

    float ms_attn_total = ms_attn_qkv + ms_qk_norm_rope + ms_attn_core + ms_attn_o;
    std::printf("  TOTAL Attention Layer (single layer)  : %7.3f ms\n", ms_attn_total);
    std::printf("  TOTAL Attention Layers (12 layers)    : %7.3f s per chunk\n\n", (ms_attn_total * 12) / 1000.0f);

    // --------------------------------------------------------------------------
    // 3. MoE Layer (48 layers)
    // --------------------------------------------------------------------------
    std::printf("--- 3. MoE Layer Breakdown (48 layers) ---\n");
    const int64_t E = 64;
    const int64_t top_k = 8;
    const int64_t inter = 640;

    // router_gemm: linear_bf16w_wmma [513 x 2560]
    std::vector<uint16_t> host_router((size_t)((E + 1) * d), 0x3c00);
    DeviceBuffer<uint16_t> w_router = DeviceBuffer<uint16_t>::from_host(host_router, "w_router");
    DeviceBuffer<float> router_logits((size_t)(T * (E + 1)), "router_logits");

    float ms_router_gemm = time_kernel([&]() {
        linear_bf16w_wmma(x_u.get(), w_router.get(), router_logits.get(), T, E + 1, d, Act::BF16, Act::F32, stream);
    }, "router_gemm (linear_bf16w_wmma [513x2560])");

    // moe_router
    DeviceBuffer<int32_t> route_ids((size_t)(T * top_k), "route_ids");
    DeviceBuffer<float> route_coef((size_t)(T * top_k), "route_coef");
    DeviceBuffer<uint32_t> router_err(1, "router_err");

    float ms_moe_router = time_kernel([&]() {
        moe_router(router_logits.get(), E + 1, T, E, top_k, router_logits.get() + E, E + 1, (int32_t)E,
                   route_ids.get(), route_coef.get(), Act::F32, router_err.get(), stream);
    }, "moe_router (top-k router)");

    // moe_group_routes
    size_t ws_bytes = moe_route_workspace_bytes(T, top_k, E);
    DeviceBuffer<uint8_t> group_ws(ws_bytes, "group_ws");

    float ms_group_routes = time_kernel([&]() {
        moe_group_routes(route_ids.get(), T, top_k, E, group_ws.get(), ws_bytes, router_err.get(), stream);
    }, "moe_group_routes");

    // Experts weights
    Q4Device w_gate_up = make_random_q4(E * 2 * inter, d, 128, rng, "gate_up");
    Q4Device w_down = make_random_q4(E * d, inter, 128, rng, "down");
    DeviceBuffer<uint16_t> h_swi((size_t)(T * top_k * inter), "h_swi");
    DeviceBuffer<float> partial((size_t)(T * top_k * d), "partial");

    float ms_gather = time_kernel([&]() {
        linear_q4_experts_gather_grouped_swiglu(x_u.get(), w_gate_up.view(), E, group_ws.get(), ws_bytes,
                                                T, top_k, h_swi.get(), Act::BF16, MoeMath::WmmaF16Wide256, stream);
    }, "gather_grouped_swiglu (Wide256)");

    float ms_combine = time_kernel([&]() {
        linear_q4_experts_combine_grouped(h_swi.get(), w_down.view(), E, route_ids.get(), route_coef.get(),
                                          group_ws.get(), ws_bytes, partial.get(), T, top_k, y_buf.get(),
                                          Act::BF16, MoeMath::WmmaF16Wide, stream);
    }, "combine_grouped (Wide)");

    // Shared expert
    Q8Device w_sh_gu = make_random_q8(2 * inter, d, 64, rng, "sh_gu");
    Q8Device w_sh_dn = make_random_q8(d, inter, 64, rng, "sh_dn");
    DeviceBuffer<uint16_t> sh_gu((size_t)(T * 2 * inter), "sh_gu");
    DeviceBuffer<uint16_t> sh_hh((size_t)(T * inter), "sh_hh");
    DeviceBuffer<uint16_t> sh_out((size_t)(T * d), "sh_out");

    float ms_shared_gu = time_kernel([&]() {
        linear_q8_wmma(x_u.get(), w_sh_gu.view(), sh_gu.get(), T, Act::BF16, Act::BF16, stream);
    }, "shared_expert gate_up (linear_q8_wmma)");

    float ms_shared_swi = time_kernel([&]() {
        swiglu(sh_gu.get(), sh_hh.get(), T, inter, Act::BF16, stream);
    }, "shared_expert swiglu");

    float ms_shared_dn = time_kernel([&]() {
        linear_q8_wmma(sh_hh.get(), w_sh_dn.view(), sh_out.get(), T, Act::BF16, Act::BF16, stream);
    }, "shared_expert down (linear_q8_wmma)");

    float ms_shared_total = ms_shared_gu + ms_shared_swi + ms_shared_dn;
    float ms_moe_total = ms_router_gemm + ms_moe_router + ms_group_routes + ms_gather + ms_combine + ms_shared_total;
    std::printf("  TOTAL Shared Expert                   : %7.3f ms\n", ms_shared_total);
    std::printf("  TOTAL MoE Layer (single layer)        : %7.3f ms\n", ms_moe_total);
    std::printf("  TOTAL MoE Layers (48 layers)          : %7.3f s per chunk\n\n", (ms_moe_total * 48) / 1000.0f);

    // --------------------------------------------------------------------------
    // 4. Hyper-Connection & Injection (48 layers x 2)
    // --------------------------------------------------------------------------
    std::printf("--- 4. Hyper-Connection & Injection Breakdown (96 calls) ---\n");
    const int64_t r = 320, R = 324;
    Q8Device w_hc_down = make_random_q8(R, n4, 64, rng, "hc_down");
    Q8ChunkMajorDevice w_hc_up = upload_random_q8_cm(n4, r, 64, rng, "hc_up");
    DeviceBuffer<uint16_t> X_buf((size_t)(T * n4), "X");
    DeviceBuffer<float> h_hc((size_t)(T * r), "h_hc");
    DeviceBuffer<float> w_in_buf((size_t)(T * H), "w_in");
    DeviceBuffer<float> inv_buf((size_t)(T * H), "inv");
    size_t hc_ws_bytes = hc_wmma_workspace_bytes(T, H, R);
    DeviceBuffer<float> hc_ws(hc_ws_bytes / 4, "hc_ws");

    float ms_hc_down = time_kernel([&]() {
        hc_mix_down_wmma(X_buf.get(), w_hc_down.view(), T, H, d, r, true, 1e-6f, h_hc.get(),
                         w_in_buf.get(), inv_buf.get(), hc_ws.get(), hc_ws_bytes, Act::BF16, stream, true);
    }, "hc_mix_down_wmma (inv_ready)");

    float ms_hc_up = time_kernel([&]() {
        hc_mix_up_wmma(h_hc.get(), w_hc_up.view(), X_buf.get(), gdn_norm.get(), inv_buf.get(),
                       T, H, d, x_u.get(), Act::BF16, stream);
    }, "hc_mix_up_wmma");

    float ms_hc_inject_inv = time_kernel([&]() {
        hc_inject_inv(X_buf.get(), w_in_buf.get(), y_buf.get(), T, H, d, 1e-6f, inv_buf.get(), Act::BF16, stream);
    }, "hc_inject_inv (fused inject + RMS)");

    float ms_hc_call = ms_hc_down + ms_hc_up + ms_hc_inject_inv;
    std::printf("  TOTAL Hyper-Connection (per call)     : %7.3f ms\n", ms_hc_call);
    std::printf("  TOTAL Hyper-Connection (96 calls)     : %7.3f s per chunk\n\n", (ms_hc_call * 96) / 1000.0f);

    // --------------------------------------------------------------------------
    // GRAND TOTAL ACCOUNTING
    // --------------------------------------------------------------------------
    float grand_total_s = (ms_gdn_total * 36 + ms_attn_total * 12 + ms_moe_total * 48 + ms_hc_call * 96) / 1000.0f;
    float estimated_tok_s = (float)T / grand_total_s;

    std::printf("================================================================================\n");
    std::printf("   TOTAL CHUNK EXECUTION TIME ACCOUNTING                                        \n");
    std::printf("================================================================================\n");
    std::printf("  36 GDN Layers Total                   : %7.3f s  (%5.1f%%)\n",
                (ms_gdn_total * 36) / 1000.0f, ((ms_gdn_total * 36) / 1000.0f) / grand_total_s * 100.0f);
    std::printf("  12 Attention Layers Total             : %7.3f s  (%5.1f%%)\n",
                (ms_attn_total * 12) / 1000.0f, ((ms_attn_total * 12) / 1000.0f) / grand_total_s * 100.0f);
    std::printf("  48 MoE Layers Total                   : %7.3f s  (%5.1f%%)\n",
                (ms_moe_total * 48) / 1000.0f, ((ms_moe_total * 48) / 1000.0f) / grand_total_s * 100.0f);
    std::printf("  96 Hyper-Connection Calls Total       : %7.3f s  (%5.1f%%)\n",
                (ms_hc_call * 96) / 1000.0f, ((ms_hc_call * 96) / 1000.0f) / grand_total_s * 100.0f);
    std::printf("--------------------------------------------------------------------------------\n");
    std::printf("  SUM OF ALL PREFILL KERNELS            : %7.3f s per 8k chunk\n", grand_total_s);
    std::printf("  ESTIMATED PREFILL THROUGHPUT          : %7.1f tok/s\n", estimated_tok_s);
    std::printf("================================================================================\n");

    (void)hipStreamDestroy(stream);
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
}

}  // namespace

int main() {
    run_breakdown(8192);
    return 0;
}
