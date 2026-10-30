// Benchmark and evaluation suite for HipKittens-inspired optimizations on AMD Strix Halo (GFX1151 / RDNA 3.5).
// Compares:
//   1. Baseline: Synchronous 8-Wave WMMA GEMM (Current Strixite wmma_gemm.hpp pattern)
//   2. Variant A: Multi-Stage Software Pipelined GEMM (Double-buffered LDS with non-draining lds_barrier)
//   3. Variant B: Producer-Consumer Wavefront Decomposition (Dedicated prefetch waves + WMMA consumer waves with s_setprio)
//   4. Variant C: XOR-Swizzled Zero-Padding LDS Tile Layout (Contiguous 16-byte chunk swizzling)

#include "common/args.hpp"
#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "common/hip_runtime.hpp"
#include "kernels/wmma.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strix;
using namespace strix::kernels;

namespace {

constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int kWmmaLdPad = BK + 8; // Current Strixite padded stride
constexpr int kWmmaLdRaw = BK;     // Unpadded stride for XOR swizzling

__device__ inline uint16_t float_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__device__ inline float bf16_to_float(uint16_t b) {
    return __uint_as_float((uint32_t)b << 16);
}

inline uint16_t host_float_to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(float));
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x007fffffu) ? 0x0040u : 0u));
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

inline float host_bf16_to_float(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

// Memory-only workgroup barrier ordering LDS without draining vector memory loads (s_waitcnt vmcnt)
__device__ __forceinline__ void lds_barrier_local() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// -------------------------------------------------------------------------------------------------
// Kernel 0: Baseline Strixite Synchronous WMMA GEMM
// 8 waves (256 threads) cooperatively load A & B into LDS, synchronize, then all waves compute WMMA
// -------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
baseline_wmma_gemm(const uint16_t * __restrict__ A, const uint16_t * __restrict__ B, float * __restrict__ C,
                   int M, int N, int K) {
    __shared__ __attribute__((aligned(16))) uint16_t As[BM * kWmmaLdPad];
    __shared__ __attribute__((aligned(16))) uint16_t Bs[BN * kWmmaLdPad];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave = tid / 32;
    const int wm = wave / 4;  // 0 or 1 (along M)
    const int wn = wave % 4;  // 0..3 (along N)

    const int m_block = by * BM;
    const int n_block = bx * BN;

    wmma_f32x8 acc[2][2] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Full coverage: 512 uint4 loads for A (2 per thread)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid * 2 + i;
            const int r = idx / 8;
            const int c = (idx % 8) * 8;
            if (m_block + r < M && k0 + c < K) {
                *reinterpret_cast<uint4 *>(&As[r * kWmmaLdPad + c]) =
                    *reinterpret_cast<const uint4 *>(&A[(m_block + r) * K + k0 + c]);
            } else {
                *reinterpret_cast<uint4 *>(&As[r * kWmmaLdPad + c]) = make_uint4(0, 0, 0, 0);
            }
        }

        // Full coverage: 1024 uint4 loads for B (4 per thread)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid * 4 + i;
            const int r = idx / 8;
            const int c = (idx % 8) * 8;
            if (n_block + r < N && k0 + c < K) {
                *reinterpret_cast<uint4 *>(&Bs[r * kWmmaLdPad + c]) =
                    *reinterpret_cast<const uint4 *>(&B[(n_block + r) * K + k0 + c]);
            } else {
                *reinterpret_cast<uint4 *>(&Bs[r * kWmmaLdPad + c]) = make_uint4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        // 4 sub-steps of 16x16 WMMA along K
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma_bf16x16 a_frag[2];
            wmma_bf16x16 b_frag[2];

            a_frag[0] = wmma_load_rows_a16(&As[(wm * 32 + 0) * kWmmaLdPad + kk], kWmmaLdPad);
            a_frag[1] = wmma_load_rows_a16(&As[(wm * 32 + 16) * kWmmaLdPad + kk], kWmmaLdPad);

            b_frag[0] = wmma_load_rows_a16(&Bs[(wn * 32 + 0) * kWmmaLdPad + kk], kWmmaLdPad);
            b_frag[1] = wmma_load_rows_a16(&Bs[(wn * 32 + 16) * kWmmaLdPad + kk], kWmmaLdPad);

            acc[0][0] = wmma_bf16(a_frag[0], b_frag[0], acc[0][0]);
            acc[0][1] = wmma_bf16(a_frag[0], b_frag[1], acc[0][1]);
            acc[1][0] = wmma_bf16(a_frag[1], b_frag[0], acc[1][0]);
            acc[1][1] = wmma_bf16(a_frag[1], b_frag[1], acc[1][1]);
        }

        __syncthreads();
    }

    // Write back results
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int r = m_block + wm * 32 + i * 16 + wmma_c_row(v);
                const int c = n_block + wn * 32 + j * 16 + (tid % 16);
                if (r < M && c < N) {
                    C[r * N + c] = acc[i][j][v];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Kernel 1: Multi-Stage Software Pipelining (Double-Buffered LDS with lds_barrier_local)
// Global loads for stage 1 are issued while computing stage 0; uses fine-grained lds_barrier_local()
// -------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
pipelined_wmma_gemm(const uint16_t * __restrict__ A, const uint16_t * __restrict__ B, float * __restrict__ C,
                    int M, int N, int K) {
    __shared__ __attribute__((aligned(16))) uint16_t As[2][BM * kWmmaLdPad];
    __shared__ __attribute__((aligned(16))) uint16_t Bs[2][BN * kWmmaLdPad];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave = tid / 32;
    const int wm = wave / 4;
    const int wn = wave % 4;

    const int m_block = by * BM;
    const int n_block = bx * BN;

    wmma_f32x8 acc[2][2] = {};

    auto load_tile = [&](int s, int k_curr) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid * 2 + i;
            const int r = idx / 8;
            const int c = (idx % 8) * 8;
            if (m_block + r < M && k_curr + c < K) {
                *reinterpret_cast<uint4 *>(&As[s][r * kWmmaLdPad + c]) =
                    *reinterpret_cast<const uint4 *>(&A[(m_block + r) * K + k_curr + c]);
            } else {
                *reinterpret_cast<uint4 *>(&As[s][r * kWmmaLdPad + c]) = make_uint4(0, 0, 0, 0);
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid * 4 + i;
            const int r = idx / 8;
            const int c = (idx % 8) * 8;
            if (n_block + r < N && k_curr + c < K) {
                *reinterpret_cast<uint4 *>(&Bs[s][r * kWmmaLdPad + c]) =
                    *reinterpret_cast<const uint4 *>(&B[(n_block + r) * K + k_curr + c]);
            } else {
                *reinterpret_cast<uint4 *>(&Bs[s][r * kWmmaLdPad + c]) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    // Prologue: Prefetch stage 0
    load_tile(0, 0);
    __syncthreads();

    int stage = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        int next_k = k0 + BK;
        int next_stage = stage ^ 1;

        if (next_k < K) {
            load_tile(next_stage, next_k);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma_bf16x16 a_frag[2];
            wmma_bf16x16 b_frag[2];

            a_frag[0] = wmma_load_rows_a16(&As[stage][(wm * 32 + 0) * kWmmaLdPad + kk], kWmmaLdPad);
            a_frag[1] = wmma_load_rows_a16(&As[stage][(wm * 32 + 16) * kWmmaLdPad + kk], kWmmaLdPad);

            b_frag[0] = wmma_load_rows_a16(&Bs[stage][(wn * 32 + 0) * kWmmaLdPad + kk], kWmmaLdPad);
            b_frag[1] = wmma_load_rows_a16(&Bs[stage][(wn * 32 + 16) * kWmmaLdPad + kk], kWmmaLdPad);

            acc[0][0] = wmma_bf16(a_frag[0], b_frag[0], acc[0][0]);
            acc[0][1] = wmma_bf16(a_frag[0], b_frag[1], acc[0][1]);
            acc[1][0] = wmma_bf16(a_frag[1], b_frag[0], acc[1][0]);
            acc[1][1] = wmma_bf16(a_frag[1], b_frag[1], acc[1][1]);
        }

        lds_barrier_local();
        stage = next_stage;
    }

    // Epilogue
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int r = m_block + wm * 32 + i * 16 + wmma_c_row(v);
                const int c = n_block + wn * 32 + j * 16 + (tid % 16);
                if (r < M && c < N) {
                    C[r * N + c] = acc[i][j][v];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Kernel 2: HipKittens Producer-Consumer Wavefront Decomposition
// 8 waves: Wave 0..1 are Producers (fetching data into double-buffered LDS)
//          Wave 2..7 are Consumers (executing WMMA at s_setprio 1)
// Synchronized via hardware s_barrier without locks or spinloops
// -------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
producer_consumer_wmma_gemm(const uint16_t * __restrict__ A, const uint16_t * __restrict__ B, float * __restrict__ C,
                            int M, int N, int K) {
    constexpr int BN_CONS = 96; // 6 consumer waves arranged as 2 x 3 (64 rows M x 96 cols N)
    __shared__ __attribute__((aligned(16))) uint16_t As[2][BM * kWmmaLdPad];
    __shared__ __attribute__((aligned(16))) uint16_t Bs[2][BN_CONS * kWmmaLdPad];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave = tid / 32;
    const int lane = tid % 32;

    const int m_block = by * BM;
    const int n_block = bx * BN_CONS;

    const bool is_producer = (wave < 2);
    const bool is_consumer = (wave >= 2);

    int tic = 0;
    int toc = 1;

    // --- PROLOGUE: Producers prefetch stage 0 ---
    if (is_producer) {
        // 64 threads load A: 512 uint4s (8 per thread)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int idx = tid * 8 + i;
            const int r = idx / 8;
            const int c = (idx % 8) * 8;
            if (m_block + r < M && c < K) {
                *reinterpret_cast<uint4 *>(&As[tic][r * kWmmaLdPad + c]) =
                    *reinterpret_cast<const uint4 *>(&A[(m_block + r) * K + c]);
            } else {
                *reinterpret_cast<uint4 *>(&As[tic][r * kWmmaLdPad + c]) = make_uint4(0, 0, 0, 0);
            }
        }

        // 64 threads load B: 768 uint4s (12 per thread)
        #pragma unroll
        for (int i = 0; i < 12; ++i) {
            const int idx = tid * 12 + i;
            const int r = idx / 8;
            const int c = (idx % 8) * 8;
            if (r < BN_CONS && n_block + r < N && c < K) {
                *reinterpret_cast<uint4 *>(&Bs[tic][r * kWmmaLdPad + c]) =
                    *reinterpret_cast<const uint4 *>(&B[(n_block + r) * K + c]);
            } else if (r < BN_CONS) {
                *reinterpret_cast<uint4 *>(&Bs[tic][r * kWmmaLdPad + c]) = make_uint4(0, 0, 0, 0);
            }
        }
        __builtin_amdgcn_s_waitcnt(0);
    }

    __syncthreads();

    // Consumer state
    const int c_wave = is_consumer ? (wave - 2) : 0; // 0..5
    const int wm = c_wave / 3;                       // 0 or 1
    const int wn = c_wave % 3;                       // 0, 1, 2
    wmma_f32x8 acc[2][2] = {};

    // --- MAIN LOOP ---
    for (int k0 = 0; k0 < K; k0 += BK, tic ^= 1, toc ^= 1) {
        const int next_k = k0 + BK;

        if (is_producer) {
            if (next_k < K) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int idx = tid * 8 + i;
                    const int r = idx / 8;
                    const int c = (idx % 8) * 8;
                    if (m_block + r < M && next_k + c < K) {
                        *reinterpret_cast<uint4 *>(&As[toc][r * kWmmaLdPad + c]) =
                            *reinterpret_cast<const uint4 *>(&A[(m_block + r) * K + next_k + c]);
                    } else {
                        *reinterpret_cast<uint4 *>(&As[toc][r * kWmmaLdPad + c]) = make_uint4(0, 0, 0, 0);
                    }
                }

                #pragma unroll
                for (int i = 0; i < 12; ++i) {
                    const int idx = tid * 12 + i;
                    const int r = idx / 8;
                    const int c = (idx % 8) * 8;
                    if (r < BN_CONS && n_block + r < N && next_k + c < K) {
                        *reinterpret_cast<uint4 *>(&Bs[toc][r * kWmmaLdPad + c]) =
                            *reinterpret_cast<const uint4 *>(&B[(n_block + r) * K + next_k + c]);
                    } else if (r < BN_CONS) {
                        *reinterpret_cast<uint4 *>(&Bs[toc][r * kWmmaLdPad + c]) = make_uint4(0, 0, 0, 0);
                    }
                }
                __builtin_amdgcn_s_waitcnt(0);
            }
        } else if (is_consumer) {
            __builtin_amdgcn_s_setprio(1); // Set high priority for compute waves

            #pragma unroll
            for (int kk = 0; kk < BK; kk += 16) {
                wmma_bf16x16 a_frag[2];
                wmma_bf16x16 b_frag[2];

                a_frag[0] = wmma_load_rows_a16(&As[tic][(wm * 32 + 0) * kWmmaLdPad + kk], kWmmaLdPad);
                a_frag[1] = wmma_load_rows_a16(&As[tic][(wm * 32 + 16) * kWmmaLdPad + kk], kWmmaLdPad);

                b_frag[0] = wmma_load_rows_a16(&Bs[tic][(wn * 32 + 0) * kWmmaLdPad + kk], kWmmaLdPad);
                b_frag[1] = wmma_load_rows_a16(&Bs[tic][(wn * 32 + 16) * kWmmaLdPad + kk], kWmmaLdPad);

                acc[0][0] = wmma_bf16(a_frag[0], b_frag[0], acc[0][0]);
                acc[0][1] = wmma_bf16(a_frag[0], b_frag[1], acc[0][1]);
                acc[1][0] = wmma_bf16(a_frag[1], b_frag[0], acc[1][0]);
                acc[1][1] = wmma_bf16(a_frag[1], b_frag[1], acc[1][1]);
            }

            __builtin_amdgcn_s_setprio(0); // Restore normal priority
        }

        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_barrier();
    }

    // Epilogue for consumers
    if (is_consumer) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    const int r = m_block + wm * 32 + i * 16 + wmma_c_row(v);
                    const int c = n_block + wn * 32 + j * 16 + (lane % 16);
                    if (r < M && c < N) {
                        C[r * N + c] = acc[i][j][v];
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Kernel 3: XOR-Swizzled Zero-Padding WMMA GEMM
// 16-byte chunk XOR swizzling eliminates LDS bank conflicts with zero padding memory waste
// -------------------------------------------------------------------------------------------------
__device__ inline int swizzle_chunk_16b(int row, int col_chunk) {
    return row * kWmmaLdRaw + ((col_chunk ^ (row & 0x7)) * 8);
}

__global__ void __launch_bounds__(256)
xorswizzled_wmma_gemm(const uint16_t * __restrict__ A, const uint16_t * __restrict__ B, float * __restrict__ C,
                      int M, int N, int K) {
    __shared__ __attribute__((aligned(16))) uint16_t As[BM * kWmmaLdRaw];
    __shared__ __attribute__((aligned(16))) uint16_t Bs[BN * kWmmaLdRaw];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave = tid / 32;
    const int wm = wave / 4;
    const int wn = wave % 4;

    const int m_block = by * BM;
    const int n_block = bx * BN;

    wmma_f32x8 acc[2][2] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Full coverage: 512 uint4 loads for A
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid * 2 + i;
            const int r = idx / 8;
            const int chunk = idx % 8;
            const int c = chunk * 8;
            if (m_block + r < M && k0 + c < K) {
                *reinterpret_cast<uint4 *>(&As[swizzle_chunk_16b(r, chunk)]) =
                    *reinterpret_cast<const uint4 *>(&A[(m_block + r) * K + k0 + c]);
            } else {
                *reinterpret_cast<uint4 *>(&As[swizzle_chunk_16b(r, chunk)]) = make_uint4(0, 0, 0, 0);
            }
        }

        // Full coverage: 1024 uint4 loads for B
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid * 4 + i;
            const int r = idx / 8;
            const int chunk = idx % 8;
            const int c = chunk * 8;
            if (n_block + r < N && k0 + c < K) {
                *reinterpret_cast<uint4 *>(&Bs[swizzle_chunk_16b(r, chunk)]) =
                    *reinterpret_cast<const uint4 *>(&B[(n_block + r) * K + k0 + c]);
            } else {
                *reinterpret_cast<uint4 *>(&Bs[swizzle_chunk_16b(r, chunk)]) = make_uint4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma_bf16x16 a_frag[2];
            wmma_bf16x16 b_frag[2];

            const int chunk0 = kk / 8;
            const int chunk1 = chunk0 + 1;

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r_a = wm * 32 + i * 16 + (tid % 16);
                const uint4 lo = *reinterpret_cast<const uint4 *>(&As[swizzle_chunk_16b(r_a, chunk0)]);
                const uint4 hi = *reinterpret_cast<const uint4 *>(&As[swizzle_chunk_16b(r_a, chunk1)]);
                const wmma_u32x8 v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                a_frag[i] = __builtin_bit_cast(wmma_bf16x16, v);
            }

            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int r_b = wn * 32 + j * 16 + (tid % 16);
                const uint4 lo = *reinterpret_cast<const uint4 *>(&Bs[swizzle_chunk_16b(r_b, chunk0)]);
                const uint4 hi = *reinterpret_cast<const uint4 *>(&Bs[swizzle_chunk_16b(r_b, chunk1)]);
                const wmma_u32x8 v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                b_frag[j] = __builtin_bit_cast(wmma_bf16x16, v);
            }

            acc[0][0] = wmma_bf16(a_frag[0], b_frag[0], acc[0][0]);
            acc[0][1] = wmma_bf16(a_frag[0], b_frag[1], acc[0][1]);
            acc[1][0] = wmma_bf16(a_frag[1], b_frag[0], acc[1][0]);
            acc[1][1] = wmma_bf16(a_frag[1], b_frag[1], acc[1][1]);
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int r = m_block + wm * 32 + i * 16 + wmma_c_row(v);
                const int c = n_block + wn * 32 + j * 16 + (tid % 16);
                if (r < M && c < N) {
                    C[r * N + c] = acc[i][j][v];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Kernel 4: Pipelined + XOR-Swizzled Zero-Pad WMMA GEMM
// Combines non-draining lds_barrier_local() double-buffering with unpadded XOR swizzling (48 KB LDS)
// -------------------------------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
pipelined_xorswizzled_wmma_gemm(const uint16_t * __restrict__ A, const uint16_t * __restrict__ B, float * __restrict__ C,
                                int M, int N, int K) {
    __shared__ __attribute__((aligned(16))) uint16_t As[2][BM * kWmmaLdRaw];
    __shared__ __attribute__((aligned(16))) uint16_t Bs[2][BN * kWmmaLdRaw];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave = tid / 32;
    const int wm = wave / 4;
    const int wn = wave % 4;

    const int m_block = by * BM;
    const int n_block = bx * BN;

    wmma_f32x8 acc[2][2] = {};

    auto load_tile_swizzled = [&](int s, int k_curr) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid * 2 + i;
            const int r = idx / 8;
            const int chunk = idx % 8;
            const int c = chunk * 8;
            if (m_block + r < M && k_curr + c < K) {
                *reinterpret_cast<uint4 *>(&As[s][swizzle_chunk_16b(r, chunk)]) =
                    *reinterpret_cast<const uint4 *>(&A[(m_block + r) * K + k_curr + c]);
            } else {
                *reinterpret_cast<uint4 *>(&As[s][swizzle_chunk_16b(r, chunk)]) = make_uint4(0, 0, 0, 0);
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid * 4 + i;
            const int r = idx / 8;
            const int chunk = idx % 8;
            const int c = chunk * 8;
            if (n_block + r < N && k_curr + c < K) {
                *reinterpret_cast<uint4 *>(&Bs[s][swizzle_chunk_16b(r, chunk)]) =
                    *reinterpret_cast<const uint4 *>(&B[(n_block + r) * K + k_curr + c]);
            } else {
                *reinterpret_cast<uint4 *>(&Bs[s][swizzle_chunk_16b(r, chunk)]) = make_uint4(0, 0, 0, 0);
            }
        }
    };

    // Prologue: Prefetch stage 0
    load_tile_swizzled(0, 0);
    __syncthreads();

    int stage = 0;
    for (int k0 = 0; k0 < K; k0 += BK) {
        int next_k = k0 + BK;
        int next_stage = stage ^ 1;

        if (next_k < K) {
            load_tile_swizzled(next_stage, next_k);
        }

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma_bf16x16 a_frag[2];
            wmma_bf16x16 b_frag[2];

            const int chunk0 = kk / 8;
            const int chunk1 = chunk0 + 1;

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r_a = wm * 32 + i * 16 + (tid % 16);
                const uint4 lo = *reinterpret_cast<const uint4 *>(&As[stage][swizzle_chunk_16b(r_a, chunk0)]);
                const uint4 hi = *reinterpret_cast<const uint4 *>(&As[stage][swizzle_chunk_16b(r_a, chunk1)]);
                const wmma_u32x8 v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                a_frag[i] = __builtin_bit_cast(wmma_bf16x16, v);
            }

            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int r_b = wn * 32 + j * 16 + (tid % 16);
                const uint4 lo = *reinterpret_cast<const uint4 *>(&Bs[stage][swizzle_chunk_16b(r_b, chunk0)]);
                const uint4 hi = *reinterpret_cast<const uint4 *>(&Bs[stage][swizzle_chunk_16b(r_b, chunk1)]);
                const wmma_u32x8 v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                b_frag[j] = __builtin_bit_cast(wmma_bf16x16, v);
            }

            acc[0][0] = wmma_bf16(a_frag[0], b_frag[0], acc[0][0]);
            acc[0][1] = wmma_bf16(a_frag[0], b_frag[1], acc[0][1]);
            acc[1][0] = wmma_bf16(a_frag[1], b_frag[0], acc[1][0]);
            acc[1][1] = wmma_bf16(a_frag[1], b_frag[1], acc[1][1]);
        }

        lds_barrier_local();
        stage = next_stage;
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int r = m_block + wm * 32 + i * 16 + wmma_c_row(v);
                const int c = n_block + wn * 32 + j * 16 + (tid % 16);
                if (r < M && c < N) {
                    C[r * N + c] = acc[i][j][v];
                }
            }
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Kernel 5: 32-Byte Contiguous Block Swizzled WMMA GEMM
// Swizzles 32-byte tiles (row & 3) so each 16x16 WMMA fragment (32 bytes) is contiguous in LDS
// Saves 50% of LDS address calculations in the inner compute loop
// -------------------------------------------------------------------------------------------------
__device__ inline int swizzle_32b_frag(int row, int blk32) {
    return row * kWmmaLdRaw + ((blk32 ^ (row & 0x3)) * 16);
}

__device__ inline int swizzle_32b_half(int row, int h) {
    return row * kWmmaLdRaw + (((h >> 1) ^ (row & 0x3)) * 16) + ((h & 1) * 8);
}

__global__ void __launch_bounds__(256)
xorswizzled32_wmma_gemm(const uint16_t * __restrict__ A, const uint16_t * __restrict__ B, float * __restrict__ C,
                        int M, int N, int K) {
    __shared__ __attribute__((aligned(16))) uint16_t As[BM * kWmmaLdRaw];
    __shared__ __attribute__((aligned(16))) uint16_t Bs[BN * kWmmaLdRaw];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tid = threadIdx.x;
    const int wave = tid / 32;
    const int wm = wave / 4;
    const int wn = wave % 4;

    const int m_block = by * BM;
    const int n_block = bx * BN;

    wmma_f32x8 acc[2][2] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Store 512 uint4 loads for A
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int idx = tid * 2 + i;
            const int r = idx / 8;
            const int h = idx % 8;
            const int c = h * 8;
            if (m_block + r < M && k0 + c < K) {
                *reinterpret_cast<uint4 *>(&As[swizzle_32b_half(r, h)]) =
                    *reinterpret_cast<const uint4 *>(&A[(m_block + r) * K + k0 + c]);
            } else {
                *reinterpret_cast<uint4 *>(&As[swizzle_32b_half(r, h)]) = make_uint4(0, 0, 0, 0);
            }
        }

        // Store 1024 uint4 loads for B
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = tid * 4 + i;
            const int r = idx / 8;
            const int h = idx % 8;
            const int c = h * 8;
            if (n_block + r < N && k0 + c < K) {
                *reinterpret_cast<uint4 *>(&Bs[swizzle_32b_half(r, h)]) =
                    *reinterpret_cast<const uint4 *>(&B[(n_block + r) * K + k0 + c]);
            } else {
                *reinterpret_cast<uint4 *>(&Bs[swizzle_32b_half(r, h)]) = make_uint4(0, 0, 0, 0);
            }
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma_bf16x16 a_frag[2];
            wmma_bf16x16 b_frag[2];

            const int blk = kk / 16;

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r_a = wm * 32 + i * 16 + (tid % 16);
                const uint4 *ptr = reinterpret_cast<const uint4 *>(&As[swizzle_32b_frag(r_a, blk)]);
                const uint4 lo = ptr[0];
                const uint4 hi = ptr[1];
                const wmma_u32x8 v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                a_frag[i] = __builtin_bit_cast(wmma_bf16x16, v);
            }

            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int r_b = wn * 32 + j * 16 + (tid % 16);
                const uint4 *ptr = reinterpret_cast<const uint4 *>(&Bs[swizzle_32b_frag(r_b, blk)]);
                const uint4 lo = ptr[0];
                const uint4 hi = ptr[1];
                const wmma_u32x8 v = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                b_frag[j] = __builtin_bit_cast(wmma_bf16x16, v);
            }

            acc[0][0] = wmma_bf16(a_frag[0], b_frag[0], acc[0][0]);
            acc[0][1] = wmma_bf16(a_frag[0], b_frag[1], acc[0][1]);
            acc[1][0] = wmma_bf16(a_frag[1], b_frag[0], acc[1][0]);
            acc[1][1] = wmma_bf16(a_frag[1], b_frag[1], acc[1][1]);
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            #pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int r = m_block + wm * 32 + i * 16 + wmma_c_row(v);
                const int c = n_block + wn * 32 + j * 16 + (tid % 16);
                if (r < M && c < N) {
                    C[r * N + c] = acc[i][j][v];
                }
            }
        }
    }
}

// Fast host reference GEMM (calculated on small subset to avoid long CPU delays)
void host_gemm_ref_fast(const uint16_t *A, const uint16_t *B, float *C, int M, int N, int K) {
    int check_m = M < 16 ? M : 16;
    int check_n = N < 96 ? N : 96; // 96 so it covers both BN=128 and BN=96
    for (int m = 0; m < check_m; ++m) {
        for (int n = 0; n < check_n; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                float a = host_bf16_to_float(A[m * K + k]);
                float b = host_bf16_to_float(B[n * K + k]);
                sum += a * b;
            }
            C[m * N + n] = sum;
        }
    }
}

struct BenchResult {
    std::string name;
    double us = 0;
    double tflops = 0;
    double gbs = 0;
    double max_diff = 0;
    bool passed = false;
};

template <typename KernelFn>
BenchResult run_benchmark(const char *name, KernelFn kernel, dim3 grid, dim3 block,
                          const uint16_t *dA, const uint16_t *dB, float *dC, float *hC,
                          const float *hRef, int M, int N, int K, int warmup, int iters) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    STRIX_HIP_CHECK(hipDeviceSynchronize(), "warmup sync");

    // Timing
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    STRIX_HIP_CHECK(hipDeviceSynchronize(), "bench sync");
    auto t1 = std::chrono::steady_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double us = total_us / iters;
    double ops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (ops / (us * 1e-6)) / 1e12;
    double bytes = ((double)M * K * 2 + (double)N * K * 2 + (double)M * N * 4);
    double gbs = (bytes / (us * 1e-6)) / 1e9;

    // Verify on subset
    STRIX_HIP_CHECK(hipMemcpyDtoH(hC, dC, (size_t)M * N * sizeof(float)), "copy C to host");
    int check_m = M < 16 ? M : 16;
    int check_n = N < 96 ? N : 96;
    double max_diff = 0;
    for (int m = 0; m < check_m; ++m) {
        for (int n = 0; n < check_n; ++n) {
            float c_val = hC[m * N + n];
            float r_val = hRef[m * N + n];
            if (std::isnan(c_val) || std::isinf(c_val)) {
                max_diff = INFINITY;
                break;
            }
            double diff = std::fabs((double)c_val - (double)r_val);
            if (diff > max_diff) max_diff = diff;
        }
    }

    bool passed = (max_diff <= 0.05 * std::sqrt((double)K)); // BF16 arithmetic tolerance
    return {name, us, tflops, gbs, max_diff, passed};
}

} // namespace

int main(int argc, char **argv) {
    int dev = 0;
    STRIX_HIP_CHECK(hipSetDevice(dev), "set device");
    hipDeviceProp_t prop{};
    STRIX_HIP_CHECK(hipGetDeviceProperties(&prop, dev), "get properties");

    printf("========================================================================\n");
    printf("   STRIXKITTENS: WMMA MATRIX PIPELINE BENCHMARK (GFX1151 / RDNA 3.5)    \n");
    printf("========================================================================\n");
    printf("Device: %s (%s, %d CUs, %zu KB LDS/WGP)\n",
           prop.name, prop.gcnArchName, prop.multiProcessorCount, prop.sharedMemPerBlock / 1024);

    struct Shape { int M, N, K; const char *desc; };
    std::vector<Shape> shapes = {
        {16, 4096, 2048, "Speculative Drafting / Batched Verify (M=16, N=4096, K=2048)"},
        {32, 4096, 2048, "Speculative Drafting / Medium Batch (M=32, N=4096, K=2048)"},
        {64, 4096, 2048, "Prefill / Prompt Chunk (M=64, N=4096, K=2048)"},
        {128, 4096, 4096, "Deep GEMM / Dense Projection (M=128, N=4096, K=4096)"}
    };

    const int warmup = 10;
    const int iters = 100;

    for (const auto &s : shapes) {
        int M = s.M, N = s.N, K = s.K;
        printf("\n------------------------------------------------------------------------\n");
        printf("Workload: %s\n", s.desc);
        printf("Shape: M=%d, N=%d, K=%d | FLOPs: %.2f MFLOP\n", M, N, K, (2.0 * M * N * K) / 1e6);

        std::vector<uint16_t> hA(M * K);
        std::vector<uint16_t> hB(N * K);
        std::vector<float> hC(M * N, 0.0f);
        std::vector<float> hRef(M * N, 0.0f);

        // Fill with normalized random values
        for (int i = 0; i < M * K; ++i) hA[i] = host_float_to_bf16((float)(rand() % 100 - 50) / 100.0f);
        for (int i = 0; i < N * K; ++i) hB[i] = host_float_to_bf16((float)(rand() % 100 - 50) / 100.0f);

        // Fast host Reference
        host_gemm_ref_fast(hA.data(), hB.data(), hRef.data(), M, N, K);

        uint16_t *dA = nullptr, *dB = nullptr;
        float *dC = nullptr;
        STRIX_HIP_CHECK(hipMalloc(&dA, M * K * sizeof(uint16_t)), "alloc dA");
        STRIX_HIP_CHECK(hipMalloc(&dB, N * K * sizeof(uint16_t)), "alloc dB");
        STRIX_HIP_CHECK(hipMalloc(&dC, M * N * sizeof(float)), "alloc dC");

        STRIX_HIP_CHECK(hipMemcpyHtoD(dA, hA.data(), M * K * sizeof(uint16_t)), "copy A");
        STRIX_HIP_CHECK(hipMemcpyHtoD(dB, hB.data(), N * K * sizeof(uint16_t)), "copy B");

        dim3 block(256);
        dim3 grid_std((N + BN - 1) / BN, (M + BM - 1) / BM);
        dim3 grid_prod_cons((N + 96 - 1) / 96, (M + BM - 1) / BM);

        std::vector<BenchResult> results;

        // 1. Baseline
        results.push_back(run_benchmark("0. Baseline Sync WMMA", baseline_wmma_gemm,
                                        grid_std, block, dA, dB, dC, hC.data(), hRef.data(),
                                        M, N, K, warmup, iters));

        // 2. Pipelined Double-buffered
        results.push_back(run_benchmark("1. Multi-Stage Pipelined", pipelined_wmma_gemm,
                                        grid_std, block, dA, dB, dC, hC.data(), hRef.data(),
                                        M, N, K, warmup, iters));

        // 3. Producer-Consumer
        results.push_back(run_benchmark("2. Producer-Consumer Waves", producer_consumer_wmma_gemm,
                                        grid_prod_cons, block, dA, dB, dC, hC.data(), hRef.data(),
                                        M, N, K, warmup, iters));

        // 4. XOR Swizzled
        results.push_back(run_benchmark("3. XOR-Swizzled LDS", xorswizzled_wmma_gemm,
                                        grid_std, block, dA, dB, dC, hC.data(), hRef.data(),
                                        M, N, K, warmup, iters));

        // 5. Hybrid: Pipelined + XOR Swizzled
        results.push_back(run_benchmark("4. Pipelined + XOR Swizzled", pipelined_xorswizzled_wmma_gemm,
                                        grid_std, block, dA, dB, dC, hC.data(), hRef.data(),
                                        M, N, K, warmup, iters));

        // 6. 32B-Block Swizzled LDS
        results.push_back(run_benchmark("5. 32B-Block Swizzled LDS", xorswizzled32_wmma_gemm,
                                        grid_std, block, dA, dB, dC, hC.data(), hRef.data(),
                                        M, N, K, warmup, iters));

        double base_us = results[0].us;
        printf("%-30s | %10s | %10s | %10s | %10s | %8s\n",
               "Kernel Implementation", "Latency", "TFLOPS", "GB/s", "Speedup", "Status");
        printf("----------------------------------------------------------------------------------------\n");
        for (const auto &r : results) {
            double speedup = base_us / r.us;
            printf("%-30s | %7.2f us | %7.2f TF | %7.2f GB | %9.2fx | %s (diff: %.4f)\n",
                   r.name.c_str(), r.us, r.tflops, r.gbs, speedup,
                   r.passed ? "PASSED" : "FAILED", r.max_diff);
        }

        STRIX_HIP_CHECK(hipFree(dA), "free dA");
        STRIX_HIP_CHECK(hipFree(dB), "free dB");
        STRIX_HIP_CHECK(hipFree(dC), "free dC");
    }

    printf("\n========================================================================\n");
    printf("   BENCHMARK SUITE COMPLETED                                            \n");
    printf("========================================================================\n");
    return 0;
}
