// Standalone test and bit-identity validator for PagedAttention (PagedReader vs PlainReader).

#include "common/check.hpp"
#include "common/hip_runtime.hpp"
#include "kernels/attention.hpp"
#include "kernels/norm.hpp"
#include "kernels/qsa.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/page_pool.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t lsb = (x >> 16) & 1;
    x += 0x7fff + lsb;
    return static_cast<uint16_t>(x >> 16);
}

float bf16_to_f32(uint16_t b) {
    uint32_t x = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

void test_block_manager_and_page_pool() {
    printf("=== Testing BlockManager & PagePool ===\n");
    // 1. Basic allocation & free
    {
        BlockManager bm(10);
        STRIX_CHECK(bm.total_blocks() == 10, "bm total");
        STRIX_CHECK(bm.free_blocks() == 10, "bm free 10");
        STRIX_CHECK(bm.used_blocks() == 0, "bm used 0");

        int32_t b0 = bm.allocate_block();
        STRIX_CHECK(b0 == 0, "b0 == 0");
        STRIX_CHECK(bm.free_blocks() == 9, "bm free 9");

        std::vector<int32_t> blocks;
        bool ok = bm.allocate_blocks(5, blocks);
        STRIX_CHECK(ok && blocks.size() == 5, "alloc 5");
        STRIX_CHECK(bm.free_blocks() == 4, "bm free 4");

        // Reference count / CoW test:
        bm.acquire_block(blocks[0]); // ref count = 2
        bm.free_block(blocks[0]);    // ref count = 1 -> should NOT be added to free list
        STRIX_CHECK(bm.free_blocks() == 4, "bm free still 4 after 1 release");
        bm.free_block(blocks[0]);    // ref count = 0 -> freed!
        STRIX_CHECK(bm.free_blocks() == 5, "bm free 5 after 2nd release");

        // Free rest:
        bm.free_block(b0);
        bm.free_blocks(blocks.data() + 1, 4);
        STRIX_CHECK(bm.free_blocks() == 10, "bm free all 10");
    }

    // 2. Multithreaded stress test
    {
        BlockManager bm(1000);
        constexpr int kThreads = 8;
        constexpr int kOpsPerThread = 500;
        std::vector<std::thread> threads;
        for (int t = 0; t < kThreads; ++t) {
            threads.emplace_back([&bm]() {
                for (int i = 0; i < kOpsPerThread; ++i) {
                    std::vector<int32_t> allocated;
                    bool ok = bm.allocate_blocks(5, allocated);
                    if (ok) {
                        // simulate some work
                        bm.free_blocks(allocated.data(), allocated.size());
                    }
                }
            });
        }
        for (auto &t : threads) t.join();
        STRIX_CHECK(bm.free_blocks() == 1000, "all blocks returned safely without race conditions");
    }

    // 3. PagePool allocation
    {
        std::vector<bool> is_attn = {true, false, true, false, true};
        PagePool pool(3200, 512, is_attn, Act::BF16, true);
        STRIX_CHECK(pool.total_tokens() == 3200, "pool total tokens");
        STRIX_CHECK(pool.total_blocks() == 100, "pool total blocks");
        STRIX_CHECK(pool.k_pool(0) != nullptr, "layer 0 k_pool not null");
        STRIX_CHECK(pool.v_pool(0) != nullptr, "layer 0 v_pool not null");
        STRIX_CHECK(pool.k_pool(1) == nullptr, "layer 1 (gdn) k_pool null");
        STRIX_CHECK(pool.k_mtp_pool() != nullptr, "mtp k_pool not null");
    }

    printf(">> [SUCCESS] BlockManager & PagePool tests passed! <<\n\n");
}

}  // namespace

int main(int argc, char **argv) {
    test_block_manager_and_page_pool();
    printf("=== Testing PagedAttention (PagedReader vs PlainReader) ===\n");

    const int64_t T = 1;        // Decode query
    const int64_t pos0 = 512;   // Attending to 513 keys (0..512)
    const int64_t Hq = 32;
    const int64_t Hk = 4;
    const int64_t D = 256;
    const int64_t W = Hk * D;
    const int64_t q_stride = Hq * 2 * D;
    const int64_t N = pos0 + T; // 513 tokens
    const int64_t block_size = 16;
    const int64_t num_logical_blocks = (N + block_size - 1) / block_size;
    const int64_t num_physical_blocks = num_logical_blocks * 3; // Sparsely distributed pool

    printf("Configuration:\n");
    printf("  T = %ld, pos0 = %ld, N = %ld\n", T, pos0, N);
    printf("  Hq = %ld, Hk = %ld, D = %ld, W = %ld\n", Hq, Hk, D, W);
    printf("  Page BlockSize = %ld, Logical Blocks = %ld, Physical Pool Blocks = %ld\n\n",
           block_size, num_logical_blocks, num_physical_blocks);

    hipStream_t stream;
    STRIX_HIP_CHECK(hipStreamCreate(&stream), "create stream");

    // 1. Generate random Q, K, V
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> h_q(T * q_stride);
    std::vector<uint16_t> h_k_flat(N * W);
    std::vector<uint16_t> h_v_flat(N * W);

    for (auto &v : h_q) v = f32_to_bf16(dist(gen));
    for (auto &v : h_k_flat) v = f32_to_bf16(dist(gen));
    for (auto &v : h_v_flat) v = f32_to_bf16(dist(gen));

    DeviceBuffer<uint16_t> d_q(h_q.size(), "q");
    DeviceBuffer<uint16_t> d_k_flat(h_k_flat.size(), "k_flat");
    DeviceBuffer<uint16_t> d_v_flat(h_v_flat.size(), "v_flat");

    STRIX_HIP_CHECK(hipMemcpy(d_q.get(), h_q.data(), h_q.size() * 2, hipMemcpyHostToDevice), "upload q");
    STRIX_HIP_CHECK(hipMemcpy(d_k_flat.get(), h_k_flat.data(), h_k_flat.size() * 2, hipMemcpyHostToDevice), "upload k_flat");
    STRIX_HIP_CHECK(hipMemcpy(d_v_flat.get(), h_v_flat.data(), h_v_flat.size() * 2, hipMemcpyHostToDevice), "upload v_flat");

    // 2. Build Scrambled Physical Page Pool
    std::vector<int32_t> block_table(num_logical_blocks);
    std::vector<bool> used(num_physical_blocks, false);
    for (int64_t b = 0; b < num_logical_blocks; ++b) {
        int64_t p = (b * 7 + 13) % num_physical_blocks;
        while (used[p]) p = (p + 1) % num_physical_blocks;
        used[p] = true;
        block_table[b] = (int32_t)p;
    }

    std::vector<uint16_t> h_pool_k(num_physical_blocks * block_size * W, 0);
    std::vector<uint16_t> h_pool_v(num_physical_blocks * block_size * W, 0);

    for (int64_t b = 0; b < num_logical_blocks; ++b) {
        const int32_t pb = block_table[b];
        const int64_t tok_start = b * block_size;
        const int64_t tok_count = std::min(block_size, N - tok_start);
        for (int64_t i = 0; i < tok_count; ++i) {
            std::memcpy(h_pool_k.data() + (pb * block_size + i) * W,
                        h_k_flat.data() + (tok_start + i) * W,
                        W * 2);
            std::memcpy(h_pool_v.data() + (pb * block_size + i) * W,
                        h_v_flat.data() + (tok_start + i) * W,
                        W * 2);
        }
    }

    DeviceBuffer<uint16_t> d_pool_k(h_pool_k.size(), "pool_k");
    DeviceBuffer<uint16_t> d_pool_v(h_pool_v.size(), "pool_v");
    DeviceBuffer<int32_t> d_block_table(block_table.size(), "block_table");

    STRIX_HIP_CHECK(hipMemcpy(d_pool_k.get(), h_pool_k.data(), h_pool_k.size() * 2, hipMemcpyHostToDevice), "upload pool_k");
    STRIX_HIP_CHECK(hipMemcpy(d_pool_v.get(), h_pool_v.data(), h_pool_v.size() * 2, hipMemcpyHostToDevice), "upload pool_v");
    STRIX_HIP_CHECK(hipMemcpy(d_block_table.get(), block_table.data(), block_table.size() * 4, hipMemcpyHostToDevice), "upload block_table");

    // 3. Attention execution
    const AttentionShape s{T, pos0, Hq, Hk, D, q_stride, 2 * D};
    const float scale = 1.0f / std::sqrt((float)D);
    const size_t ws_bytes = attention_workspace_bytes(s);
    DeviceBuffer<float> d_ws(ws_bytes / 4 + 1, "workspace");

    DeviceBuffer<uint16_t> d_out_plain(T * Hq * D, "out_plain");
    DeviceBuffer<uint16_t> d_out_paged(T * Hq * D, "out_paged");

    const void *gate = d_q.get() + (size_t)D;

    // Run Plain:
    attention(d_q.get(), gate, d_k_flat.get(), d_v_flat.get(), N, s, scale,
              d_out_plain.get(), d_ws.get(), ws_bytes, Act::BF16, stream);

    // Run Paged:
    PagedKvView paged_view{
        d_pool_k.get(),
        d_pool_v.get(),
        d_block_table.get(),
        N,
        W,
        block_size
    };
    attention(d_q.get(), gate, paged_view, s, scale,
              d_out_paged.get(), d_ws.get(), ws_bytes, Act::BF16, stream);

    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync");

    // 4. Verify bit-exact identity
    std::vector<uint16_t> h_out_plain(T * Hq * D);
    std::vector<uint16_t> h_out_paged(T * Hq * D);

    STRIX_HIP_CHECK(hipMemcpy(h_out_plain.data(), d_out_plain.get(), h_out_plain.size() * 2, hipMemcpyDeviceToHost), "read out_plain");
    STRIX_HIP_CHECK(hipMemcpy(h_out_paged.data(), d_out_paged.get(), h_out_paged.size() * 2, hipMemcpyDeviceToHost), "read out_paged");

    float max_diff = 0.0f;
    int mismatches = 0;
    for (size_t i = 0; i < h_out_plain.size(); ++i) {
        float p1 = bf16_to_f32(h_out_plain[i]);
        float p2 = bf16_to_f32(h_out_paged[i]);
        float diff = std::abs(p1 - p2);
        if (diff > max_diff) max_diff = diff;
        if (h_out_plain[i] != h_out_paged[i]) mismatches++;
    }

    printf("=== Dense Attention Test Result ===\n");
    printf("Max absolute difference: %f\n", max_diff);
    printf("Mismatched uint16 words: %d / %zu\n", mismatches, h_out_plain.size());
    if (mismatches == 0 && max_diff == 0.0f) {
        printf(">> [SUCCESS] PagedReader is 100%% BIT-IDENTICAL to PlainReader (Dense)! <<\n\n");
    } else {
        printf(">> [FAILURE] Mismatch detected in Dense Attention! <<\n\n");
        return 1;
    }

    // 4b. Test QSA Gathered Attention
    printf("=== Testing QSA Gathered Attention (PagedReader vs PlainReader) ===\n");
    const int64_t qsa_k = 32; // select 32 QSA blocks
    const int64_t complete_blocks = (pos0 + T) / kQsaBlock;
    std::vector<int32_t> h_sel(T * qsa_k);
    std::vector<int32_t> h_nsel(T, (int32_t)std::min(qsa_k, complete_blocks));
    for (int64_t b = 0; b < h_nsel[0]; ++b) {
        h_sel[b] = (int32_t)(b * 2 % complete_blocks); // arbitrary valid selection
    }

    DeviceBuffer<int32_t> d_sel(h_sel.size(), "sel");
    DeviceBuffer<int32_t> d_nsel(h_nsel.size(), "nsel");
    DeviceBuffer<uint32_t> d_err(3, "err");
    STRIX_HIP_CHECK(hipMemcpy(d_sel.get(), h_sel.data(), h_sel.size() * 4, hipMemcpyHostToDevice), "upload sel");
    STRIX_HIP_CHECK(hipMemcpy(d_nsel.get(), h_nsel.data(), h_nsel.size() * 4, hipMemcpyHostToDevice), "upload nsel");
    STRIX_HIP_CHECK(hipMemset(d_err.get(), 0, 12), "zero err");

    const size_t gws_bytes = attention_gathered_workspace_bytes(s, qsa_k);
    DeviceBuffer<float> d_gws(gws_bytes / 4 + 1, "g_workspace");
    DeviceBuffer<uint16_t> d_gout_plain(T * Hq * D, "gout_plain");
    DeviceBuffer<uint16_t> d_gout_paged(T * Hq * D, "gout_paged");

    attention_gathered(d_q.get(), gate, d_k_flat.get(), d_v_flat.get(), N, s,
                       d_sel.get(), d_nsel.get(), qsa_k, scale, d_gout_plain.get(),
                       d_gws.get(), gws_bytes, d_err.get(), Act::BF16, stream);

    attention_gathered(d_q.get(), gate, paged_view, s,
                       d_sel.get(), d_nsel.get(), qsa_k, scale, d_gout_paged.get(),
                       d_gws.get(), gws_bytes, d_err.get(), Act::BF16, stream);

    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync gathered");

    std::vector<uint16_t> h_gout_plain(T * Hq * D);
    std::vector<uint16_t> h_gout_paged(T * Hq * D);
    STRIX_HIP_CHECK(hipMemcpy(h_gout_plain.data(), d_gout_plain.get(), h_gout_plain.size() * 2, hipMemcpyDeviceToHost), "read gout_plain");
    STRIX_HIP_CHECK(hipMemcpy(h_gout_paged.data(), d_gout_paged.get(), h_gout_paged.size() * 2, hipMemcpyDeviceToHost), "read gout_paged");

    float g_max_diff = 0.0f;
    int g_mismatches = 0;
    for (size_t i = 0; i < h_gout_plain.size(); ++i) {
        float p1 = bf16_to_f32(h_gout_plain[i]);
        float p2 = bf16_to_f32(h_gout_paged[i]);
        float diff = std::abs(p1 - p2);
        if (diff > g_max_diff) g_max_diff = diff;
        if (h_gout_plain[i] != h_gout_paged[i]) g_mismatches++;
    }

    printf("QSA Gathered max absolute difference: %f\n", g_max_diff);
    printf("QSA Gathered mismatched uint16 words: %d / %zu\n", g_mismatches, h_gout_plain.size());
    if (g_mismatches == 0 && g_max_diff == 0.0f) {
        printf(">> [SUCCESS] PagedReader is 100%% BIT-IDENTICAL in QSA Gathered Attention! <<\n\n");
    } else {
        printf(">> [FAILURE] Mismatch in QSA Gathered Attention! <<\n\n");
        return 1;
    }

    // 4c. Test with BlockSize = 32
    printf("=== Testing BlockSize = 32 (PagedReader vs PlainReader) ===\n");
    const int64_t bs32 = 32;
    const int64_t num_logical_blocks32 = (N + bs32 - 1) / bs32;
    const int64_t num_physical_blocks32 = num_logical_blocks32 * 3;

    std::vector<int32_t> bt32(num_logical_blocks32);
    std::vector<bool> used32(num_physical_blocks32, false);
    for (int64_t b = 0; b < num_logical_blocks32; ++b) {
        int64_t p = (b * 11 + 5) % num_physical_blocks32;
        while (used32[p]) p = (p + 1) % num_physical_blocks32;
        used32[p] = true;
        bt32[b] = (int32_t)p;
    }

    std::vector<uint16_t> h_pool_k32(num_physical_blocks32 * bs32 * W, 0);
    std::vector<uint16_t> h_pool_v32(num_physical_blocks32 * bs32 * W, 0);
    for (int64_t b = 0; b < num_logical_blocks32; ++b) {
        const int32_t pb = bt32[b];
        const int64_t tok_start = b * bs32;
        const int64_t tok_count = std::min(bs32, N - tok_start);
        for (int64_t i = 0; i < tok_count; ++i) {
            std::memcpy(h_pool_k32.data() + (pb * bs32 + i) * W, h_k_flat.data() + (tok_start + i) * W, W * 2);
            std::memcpy(h_pool_v32.data() + (pb * bs32 + i) * W, h_v_flat.data() + (tok_start + i) * W, W * 2);
        }
    }

    DeviceBuffer<uint16_t> d_pool_k32(h_pool_k32.size(), "pool_k32");
    DeviceBuffer<uint16_t> d_pool_v32(h_pool_v32.size(), "pool_v32");
    DeviceBuffer<int32_t> d_bt32(bt32.size(), "bt32");
    DeviceBuffer<uint16_t> d_out32(T * Hq * D, "out32");

    STRIX_HIP_CHECK(hipMemcpy(d_pool_k32.get(), h_pool_k32.data(), h_pool_k32.size() * 2, hipMemcpyHostToDevice), "upload pool_k32");
    STRIX_HIP_CHECK(hipMemcpy(d_pool_v32.get(), h_pool_v32.data(), h_pool_v32.size() * 2, hipMemcpyHostToDevice), "upload pool_v32");
    STRIX_HIP_CHECK(hipMemcpy(d_bt32.get(), bt32.data(), bt32.size() * 4, hipMemcpyHostToDevice), "upload bt32");

    PagedKvView paged_view32{d_pool_k32.get(), d_pool_v32.get(), d_bt32.get(), N, W, bs32};
    attention(d_q.get(), gate, paged_view32, s, scale, d_out32.get(), d_ws.get(), ws_bytes, Act::BF16, stream);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync bs32");

    std::vector<uint16_t> h_out32(T * Hq * D);
    STRIX_HIP_CHECK(hipMemcpy(h_out32.data(), d_out32.get(), h_out32.size() * 2, hipMemcpyDeviceToHost), "read out32");

    float max_diff32 = 0.0f;
    int mismatches32 = 0;
    for (size_t i = 0; i < h_out_plain.size(); ++i) {
        float p1 = bf16_to_f32(h_out_plain[i]);
        float p2 = bf16_to_f32(h_out32[i]);
        float diff = std::abs(p1 - p2);
        if (diff > max_diff32) max_diff32 = diff;
        if (h_out_plain[i] != h_out32[i]) mismatches32++;
    }

    printf("BlockSize=32 max absolute difference: %f\n", max_diff32);
    printf("BlockSize=32 mismatched uint16 words: %d / %zu\n", mismatches32, h_out32.size());
    if (mismatches32 == 0 && max_diff32 == 0.0f) {
        printf(">> [SUCCESS] PagedReader (BlockSize=32) is 100%% BIT-IDENTICAL! <<\n\n");
    } else {
        printf(">> [FAILURE] Mismatch in BlockSize=32! <<\n\n");
        return 1;
    }

    // 4b. Test paged_kv_scatter
    printf("=== Testing paged_kv_scatter ===\n");
    const int64_t scatter_T = 128;
    const int64_t scatter_pos0 = 37; // non-aligned starting position
    std::vector<uint16_t> h_k_scatter(scatter_T * W), h_v_scatter(scatter_T * W);
    for (size_t i = 0; i < h_k_scatter.size(); ++i) {
        h_k_scatter[i] = static_cast<uint16_t>(0x3000 + (i & 0x0FFF));
        h_v_scatter[i] = static_cast<uint16_t>(0x4000 + (i & 0x0FFF));
    }
    DeviceBuffer<uint16_t> d_k_scatter(h_k_scatter.size(), "k_scatter");
    DeviceBuffer<uint16_t> d_v_scatter(h_v_scatter.size(), "v_scatter");
    STRIX_HIP_CHECK(hipMemcpy(d_k_scatter.get(), h_k_scatter.data(), h_k_scatter.size() * 2, hipMemcpyHostToDevice), "up k_scat");
    STRIX_HIP_CHECK(hipMemcpy(d_v_scatter.get(), h_v_scatter.data(), h_v_scatter.size() * 2, hipMemcpyHostToDevice), "up v_scat");

    paged_kv_scatter(d_k_scatter.get(), d_v_scatter.get(), W,
                     d_pool_k32.get(), d_pool_v32.get(), d_bt32.get(),
                     scatter_T, scatter_pos0, bs32, W, Act::BF16, stream);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "scatter sync");

    std::vector<uint16_t> h_pool_k_check(num_physical_blocks32 * bs32 * W);
    std::vector<uint16_t> h_pool_v_check(num_physical_blocks32 * bs32 * W);
    STRIX_HIP_CHECK(hipMemcpy(h_pool_k_check.data(), d_pool_k32.get(), h_pool_k_check.size() * 2, hipMemcpyDeviceToHost), "down k_pool");
    STRIX_HIP_CHECK(hipMemcpy(h_pool_v_check.data(), d_pool_v32.get(), h_pool_v_check.size() * 2, hipMemcpyDeviceToHost), "down v_pool");

    int scatter_mismatches = 0;
    for (int64_t t = 0; t < scatter_T; ++t) {
        int64_t row = scatter_pos0 + t;
        int64_t b = row / bs32;
        int64_t o = row % bs32;
        int64_t pb = bt32[b];
        int64_t phys_row = pb * bs32 + o;
        for (int64_t col = 0; col < W; ++col) {
            uint16_t exp_k = h_k_scatter[t * W + col];
            uint16_t exp_v = h_v_scatter[t * W + col];
            uint16_t act_k = h_pool_k_check[phys_row * W + col];
            uint16_t act_v = h_pool_v_check[phys_row * W + col];
            if (exp_k != act_k || exp_v != act_v) scatter_mismatches++;
        }
    }
    printf("paged_kv_scatter mismatched elements: %d / %lld\n", scatter_mismatches, (long long)(scatter_T * W * 2));
    if (scatter_mismatches == 0) {
        printf(">> [SUCCESS] paged_kv_scatter is 100%% BIT-IDENTICAL! <<\n\n");
    } else {
        printf(">> [FAILURE] paged_kv_scatter had mismatches! <<\n\n");
        return 1;
    }

    // 4c. Test Multi-Token Prefill Attention (T = 16)
    printf("=== Testing Multi-Token Prefill Attention (T = 16) ===\n");
    // Restore d_pool_k32 and d_pool_v32 from h_pool_k32 (which was mapped from h_k_flat / h_v_flat)
    STRIX_HIP_CHECK(hipMemcpy(d_pool_k32.get(), h_pool_k32.data(), h_pool_k32.size() * 2, hipMemcpyHostToDevice), "restore pool_k32");
    STRIX_HIP_CHECK(hipMemcpy(d_pool_v32.get(), h_pool_v32.data(), h_pool_v32.size() * 2, hipMemcpyHostToDevice), "restore pool_v32");

    const int64_t prefill_T = 16;
    const int64_t prefill_pos0 = 256;
    const int64_t prefill_N = prefill_pos0 + prefill_T;
    const AttentionShape s_prefill{prefill_T, prefill_pos0, Hq, Hk, D, q_stride, 2 * D};
    std::vector<uint16_t> h_q_prefill(prefill_T * q_stride);
    for (auto &v : h_q_prefill) v = f32_to_bf16(dist(gen));
    DeviceBuffer<uint16_t> d_q_prefill(h_q_prefill.size(), "q_prefill");
    STRIX_HIP_CHECK(hipMemcpy(d_q_prefill.get(), h_q_prefill.data(), h_q_prefill.size() * 2, hipMemcpyHostToDevice), "upload q_prefill");

    DeviceBuffer<uint16_t> d_out_prefill_plain(prefill_T * Hq * D, "out_prefill_plain");
    DeviceBuffer<uint16_t> d_out_prefill_paged(prefill_T * Hq * D, "out_prefill_paged");

    const size_t prefill_ws_bytes = attention_workspace_bytes(s_prefill);
    DeviceBuffer<float> d_prefill_ws(prefill_ws_bytes / 4 + 1, "prefill_ws");

    attention(d_q_prefill.get(), d_q_prefill.get() + D, d_k_flat.get(), d_v_flat.get(), prefill_N,
              s_prefill, scale, d_out_prefill_plain.get(), d_prefill_ws.get(), prefill_ws_bytes, Act::BF16, stream);
    attention(d_q_prefill.get(), d_q_prefill.get() + D, paged_view32,
              s_prefill, scale, d_out_prefill_paged.get(), d_prefill_ws.get(), prefill_ws_bytes, Act::BF16, stream);
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "prefill sync");

    std::vector<uint16_t> h_out_prefill_plain(prefill_T * Hq * D);
    std::vector<uint16_t> h_out_prefill_paged(prefill_T * Hq * D);
    STRIX_HIP_CHECK(hipMemcpy(h_out_prefill_plain.data(), d_out_prefill_plain.get(), h_out_prefill_plain.size() * 2, hipMemcpyDeviceToHost), "read prefill plain");
    STRIX_HIP_CHECK(hipMemcpy(h_out_prefill_paged.data(), d_out_prefill_paged.get(), h_out_prefill_paged.size() * 2, hipMemcpyDeviceToHost), "read prefill paged");

    float prefill_max_diff = 0.0f;
    int prefill_mismatches = 0;
    for (size_t i = 0; i < h_out_prefill_plain.size(); ++i) {
        float p1 = bf16_to_f32(h_out_prefill_plain[i]);
        float p2 = bf16_to_f32(h_out_prefill_paged[i]);
        float diff = std::abs(p1 - p2);
        if (diff > prefill_max_diff) prefill_max_diff = diff;
        if (h_out_prefill_plain[i] != h_out_prefill_paged[i]) prefill_mismatches++;
    }
    printf("Prefill (T=16) max absolute difference: %f\n", prefill_max_diff);
    printf("Prefill (T=16) mismatched elements: %d / %zu\n", prefill_mismatches, h_out_prefill_plain.size());
    if (prefill_mismatches == 0 && prefill_max_diff == 0.0f) {
        printf(">> [SUCCESS] Multi-Token Prefill Attention is 100%% BIT-IDENTICAL! <<\n\n");
    } else {
        printf(">> [FAILURE] Mismatch in Multi-Token Prefill Attention! <<\n\n");
        return 1;
    }

    // 5. Benchmark Latency
    printf("Running 200 benchmark iterations...\n");
    hipEvent_t start, stop;
    STRIX_HIP_CHECK(hipEventCreate(&start), "create start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "create stop");

    // Warmup
    for (int i = 0; i < 20; ++i) {
        attention(d_q.get(), gate, d_k_flat.get(), d_v_flat.get(), N, s, scale, d_out_plain.get(), d_ws.get(), ws_bytes, Act::BF16, stream);
        attention(d_q.get(), gate, paged_view, s, scale, d_out_paged.get(), d_ws.get(), ws_bytes, Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "warmup sync");

    STRIX_HIP_CHECK(hipEventRecord(start, stream), "record start");
    for (int i = 0; i < 200; ++i) {
        attention(d_q.get(), gate, d_k_flat.get(), d_v_flat.get(), N, s, scale, d_out_plain.get(), d_ws.get(), ws_bytes, Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipEventRecord(stop, stream), "record stop");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync plain");
    float t_plain = 0.0f;
    STRIX_HIP_CHECK(hipEventElapsedTime(&t_plain, start, stop), "time plain");

    STRIX_HIP_CHECK(hipEventRecord(start, stream), "record start");
    for (int i = 0; i < 200; ++i) {
        attention(d_q.get(), gate, paged_view, s, scale, d_out_paged.get(), d_ws.get(), ws_bytes, Act::BF16, stream);
    }
    STRIX_HIP_CHECK(hipEventRecord(stop, stream), "record stop");
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "sync paged");
    float t_paged = 0.0f;
    STRIX_HIP_CHECK(hipEventElapsedTime(&t_paged, start, stop), "time paged");

    printf("PlainReader time (200 calls): %.3f ms (%.3f us / call)\n", t_plain, t_plain * 1000.0f / 200.0f);
    printf("PagedReader time (200 calls): %.3f ms (%.3f us / call)\n", t_paged, t_paged * 1000.0f / 200.0f);
    printf("Overhead from address translation: %+.2f%%\n\n", (t_paged - t_plain) / t_plain * 100.0f);

    printf("ALL TESTS PASSED!\n");
    return 0;
}
