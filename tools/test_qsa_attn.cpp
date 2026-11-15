#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "kernels/attention.hpp"
#include "runtime/device_buffer.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace strix;
using namespace strix::kernels;

int main() {
    const int64_t T = 8192, hq = 24, hkv = 2, hd = 256, astride = 13952;
    const int64_t pos0 = 65536; // past dense limit
    const int64_t qsa_blocks = 64; // 4096 keys selected
    const int64_t capacity = 131072;

    AttentionShape as{T, pos0, hq, hkv, hd, astride, 2 * hd};
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    size_t ws_bytes = attention_gathered_wmma_workspace_bytes(as, qsa_blocks);
    DeviceBuffer<float> ws(ws_bytes / sizeof(float), "ws");

    DeviceBuffer<uint16_t> q((size_t)(T * astride), "q");
    DeviceBuffer<uint16_t> k_cache((size_t)(capacity * hkv * hd), "k_cache");
    DeviceBuffer<uint16_t> v_cache((size_t)(capacity * hkv * hd), "v_cache");
    DeviceBuffer<uint16_t> out((size_t)(T * hq * hd), "out");

    std::vector<int32_t> host_sel((size_t)(T * qsa_blocks));
    for (size_t i = 0; i < host_sel.size(); ++i) host_sel[i] = (int32_t)(i % 1000);
    std::vector<int32_t> host_nsel((size_t)T, (int32_t)qsa_blocks);

    DeviceBuffer<int32_t> sel = DeviceBuffer<int32_t>::from_host(host_sel, "sel");
    DeviceBuffer<int32_t> nsel = DeviceBuffer<int32_t>::from_host(host_nsel, "nsel");
    DeviceBuffer<uint32_t> err(4, "err");

    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr, stop = nullptr;
    STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
    STRIX_HIP_CHECK(hipEventCreate(&start), "start");
    STRIX_HIP_CHECK(hipEventCreate(&stop), "stop");

    // Warmup
    for (int i = 0; i < 2; ++i) {
        attention_gathered_wmma(q.get(), q.get() + hd, k_cache.get(), v_cache.get(), capacity,
                                as, sel.get(), nsel.get(), qsa_blocks, attn_scale, out.get(),
                                ws.get(), ws_bytes, err.get(), Act::BF16, stream, AttnPrecision::Fast);
    }
    STRIX_HIP_CHECK(hipStreamSynchronize(stream), "warmup sync");

    float total_ms = 0.0f;
    int iters = 15;
    for (int i = 0; i < iters; ++i) {
        STRIX_HIP_CHECK(hipEventRecord(start, stream), "start");
        attention_gathered_wmma(q.get(), q.get() + hd, k_cache.get(), v_cache.get(), capacity,
                                as, sel.get(), nsel.get(), qsa_blocks, attn_scale, out.get(),
                                ws.get(), ws_bytes, err.get(), Act::BF16, stream, AttnPrecision::Fast);
        STRIX_HIP_CHECK(hipEventRecord(stop, stream), "stop");
        STRIX_HIP_CHECK(hipEventSynchronize(stop), "sync");
        float ms = 0.0f;
        STRIX_HIP_CHECK(hipEventElapsedTime(&ms, start, stop), "time");
        total_ms += ms;
    }
    std::printf("attention_gathered_wmma (pos0=65k, 64 blocks, Fast): %6.3f ms\n", total_ms / iters);

    return 0;
}
