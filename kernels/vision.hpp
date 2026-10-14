#pragma once

// Vision tower kernels: the pieces
// around the BF16 GEMMs - vision_linear_bias is its own dev-stage kernel, not kernels/linear (the tower
// needs the bias added on the FP32 accumulator, with BF16 weights). Activations are F32 or BF16 (raw bits); math is
// FP32, one rounding where the output is stored. Tower weights are BF16 in vision.strixw; the norm
// gains/biases reach these kernels widened to FP32 by the loader (the engine's convention).

#include "kernels/norm.hpp"  // Act

#include "common/hip_runtime.hpp"

#include <cstdint>

namespace strix::kernels {

// y[r, c] += bias[c], rows x cols contiguous. Separate from residual_add (per-row vector, not elementwise
// equal-length); the tower adds a bias after every GEMM.
void vision_bias_add(void *y, const void *bias, int64_t rows, int64_t cols, Act act, hipStream_t stream);

// LayerNorm (mean-centred, unlike the trunk's RMSNorms): out = (x - mean) / sqrt(var + eps) * g + beta,
// over rows of d. g/beta FP32 [d]. out == x is allowed (one block per row, reads finish before writes).
void vision_layernorm(const void *x, const float *g, const float *beta, void *out, int64_t rows, int64_t d,
                      float eps, Act act, hipStream_t stream);

// The two GELUs of the tower: the block MLP's gelu_pytorch_tanh and the merger's plain (erf) GELU.
enum class VisionGelu { Tanh, Erf };
const char *vision_gelu_name(VisionGelu g);
void vision_gelu(void *x, int64_t n, VisionGelu gelu, Act act, hipStream_t stream);

// out[r, :] = in[r, :] + sum_t w[r][t] * table[idx[r][t], :] - the learned 48x48 position table
// bilinear-resampled (4 taps). table is the tower's BF16 rows of d; idx int32 [rows, taps], w FP32
// [rows, taps]. out == in is allowed (per-row, reads before writes).
void vision_pos_add(const void *in, const uint16_t *table, const int32_t *idx, const float *w, void *out,
                    int64_t rows, int64_t d, int64_t taps, Act act, hipStream_t stream);

// Axial 2D rope on the q and k slices of the fused qkv rows [n, 3*heads*hd] in place. Rows are in the
// merge-block order, so (h, w) follow from the grid arithmetically - no position table. theta is the
// checkpoint's (1e4); head_dim must be even; angles are the [h | w | h | w] recomposition (rotate_half
// pairs i, i + hd/2).
void vision_rope_qk(void *qkv, int64_t n, int64_t heads, int64_t hd, int64_t grid_h, int64_t grid_w,
                    int64_t merge, float theta, Act act, hipStream_t stream);

// Dense, non-causal attention for ONE image (one temporal segment - cu_seqlens is [0, n] for t = 1;
// video segments are a later step and must extend this). q/k/v are the head-h columns of the fused rope'd
// qkv rows [n, 3*h]; out is [n, heads*hd] with head-major columns (the layout the output-projection GEMM
// takes). One block per (row, head); FP32 scores/softmax; scale 1/sqrt(hd). n <= kMaxVisionSeq (the
// per-block shared scores array is n FP32 = 32 KiB at the cap; a 1080p screenshot is 8160). out must not
// overlap qkv (every row of qkv is still read while earlier rows' outputs are written).
constexpr int64_t kMaxVisionSeq = 8192;
// BF16 runs a one-pass WMMA kernel (online softmax, 64-row x 64-key tiles) and needs head_dim % 8 == 0, <= 80 and
// a 16-byte aligned qkv; F32 stays the per-row form above.
void vision_attn(const void *qkv, void *out, int64_t n, int64_t heads, int64_t hd, Act act, hipStream_t stream);

// y = x @ W^T + bias, W [N,K] raw BF16 bits (checkpoint layout), bias stored in the activation dtype.
// The bias is added on the FP32 accumulator before the single rounding to the output dtype - the same
// rounding points as torch's F.linear; the separate vision_bias_add keeps a double rounding (measured:
// 7% worse than the oracle's own BF16 deviation over a 27-layer tower).
// BF16 runs on the matrix cores and needs K % 16 == 0 and 16-byte aligned x and w; F32 stays the scalar form.
void vision_linear_bias(const void *x, const uint16_t *w, const void *bias, void *y, int64_t M, int64_t N,
                        int64_t K, Act act, hipStream_t stream);

// The merger's 2x2 spatial regroup: rows are in merge-block order, so m*m consecutive input rows form one
// output row: out[u, slot * d + :] = in[u * m*m + slot, :] (a copy, no arithmetic).
void vision_merge_regroup(const void *in, void *out, int64_t rows, int64_t d, int64_t merge, Act act,
                          hipStream_t stream);

// Replace rows of the embedding written into the hyper-connection streams (the same row layout as
// kernels::embedding_lookup's copies: stream copy c of target row t at x + t * out_ts + c * d): for each
// image row r (sequence order), x[row_dev[r] * out_ts + c * d + col] = merged[r * d + col] for every copy
// c < copies - the tower's merged rows take the place of the image-token embeddings in all streams.
// One block per output row (the merger rows are read-only, the target rows are distinct, so no races);
// a pure copy, bit-exact in both dtypes. row_dev: int32 device indices into [0, total_rows).
void vision_embedding_scatter(void *x, const int32_t *row_dev, int64_t rows, const void *merged, int64_t d,
                              int64_t total_rows, int64_t copies, int64_t out_ts, Act act, hipStream_t stream);

}  // namespace strix::kernels
