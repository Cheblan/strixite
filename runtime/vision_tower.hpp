#pragma once

// The vision tower inside the engine: the 0.45B ViT + merger resident on the
// device from vision.strixw, one forward per image. The checkpoint's `model.visual.*` tensors arrive
// bit-exact as BF16 rows (the converter's --vision pass, validated tensor by tensor against the
// checkpoint's config.json "vision_config"); everything the tower reads is shape-checked here at load,
// so a file that isn't the tower this code runs fails at construction, named.
//
// Dimensions come from the tensor shapes (the file carries no JSON the runtime must parse): hidden from
// the patch-embed output columns, depth by counting complete blocks, the position-table side from its row
// count. The block and merger math runs on the gated kernels of kernels/vision.hpp with the same launch
// sequence the tower tests gate against; LayerNorm gains / biases are widened to FP32 here (the engine's
// convention: math FP32, activations in the activation dtype, one rounding where an op stores).
//
// One image per forward (t = 1): the temporal grid axis must be 1 - video is not built.

#include "common/hip_runtime.hpp"
#include "formats/strixw.hpp"
#include "kernels/norm.hpp"  // Act
#include "runtime/device_buffer.hpp"
#include "runtime/strixw_loader.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strix {

// Everything vision_config implies that the tower code needs, checked against the file's shapes.
struct VisionTowerDims {
    int64_t depth = 0;      // ViT blocks
    int64_t hidden = 0;     // ViT width (1152)
    int64_t heads = 0;      // attention heads (16); head_dim = hidden / heads (72)
    int64_t inter = 0;      // block MLP width (4304)
    int64_t patch_in = 0;   // pixel_values columns: in_channels x temporal x patch x patch (3 x 2 x 16 x 16)
    int64_t merge = 0;      // spatial merge side (2)
    int64_t out_hidden = 0; // merger output width: the text hidden_size (2560)
    int64_t pos_side = 0;   // learned position table: pos_side x pos_side rows (48)
    int64_t head_dim() const { return hidden / heads; }
};

// Scratch pointers for one tower forward (the caller owns the memory - runtime/qwen4exp.cpp, the session).
// Required element counts are scratch_bytes() below; act dtype throughout except idx (int32) / w (FP32).
struct TowerScratch {
    void *pixel_values = nullptr;  // [n, patch_in]: the caller's upload
    void *x = nullptr;             // [n, hidden]: the persistent stream
    void *conv = nullptr;          // [n, hidden]
    void *ln = nullptr;            // [n, hidden]
    void *proj = nullptr;          // [n, hidden]
    void *qkv = nullptr;           // [n, 3 * hidden]
    void *mlp = nullptr;           // [n, inter]
    void *merg = nullptr;          // [n / merge^2, merge^2 * hidden] (its own: fc1 reads it while writing qkv)
    int32_t *idx = nullptr;        // [n, 4] position-table taps (grid_taps)
    float *w = nullptr;            // [n, 4] tap weights
    void *merged = nullptr;        // [n / merge^2, out_hidden]: the output
};
// The per-buffer element counts for n patches.
struct TowerScratchBytes {
    int64_t patch_in = 0, hidden = 0, qkv = 0, mlp = 0, merg = 0, merged = 0;  // hidden covers x / conv / ln / proj
};

class VisionTower {
public:
    // path: a vision.strixw from the same checkpoint as text_weights (the side-file rule the n-gram table
    // follows: kind must be "vision" and source_fingerprint must equal the text file's - a tower from
    // another checkpoint is refused naming both). The tower loads to the device whatever the text model's
    // placement is: HostExperts is a contract about expert stacks, which the tower file has none of, and
    // 856 MiB of BF16 is a device region on every machine.
    VisionTower(const std::string &path, const StrixwFile &text_weights, kernels::Act act);

    const VisionTowerDims &dims() const { return dims_; }
    kernels::Act act() const { return act_; }
    const std::string &path() const { return path_; }

    // The bilinear resample of the learned pos_side x pos_side position table to an (grid_h, grid_w) patch
    // grid: align_corners (src = i * (side - 1) / (size - 1)), 4 taps, border-clamped indices, separable
    // outer-product weights, rows in the merge-block order the patches and the merger use. idx into the
    // flattened table (row-major side x side) and their FP32 weights, n = grid_h * grid_w rows of 4.
    void grid_taps(int64_t grid_h, int64_t grid_w, std::vector<int32_t> &idx, std::vector<float> &w) const;

    // n = grid_h * grid_w patches (n % merge^2 == 0, n >= merge^2, n <= the attention cap); scratch: the
    // TowerScratchBytes(n)-sized buffers, every one distinct (the kernels' in-place rules are stricter than
    // the sizes need - keep them separate). merged gets the n / merge^2 rows of out_hidden the trunk
    // scatters at the image positions. The stream is synchronized by the caller, not here.
    TowerScratchBytes scratch_bytes(int64_t grid_h, int64_t grid_w) const;
    void forward(int64_t grid_h, int64_t grid_w, const int32_t *idx_dev, const float *w_dev,
                 const TowerScratch &scratch, hipStream_t stream) const;

private:
    struct PerBlock {
        const uint16_t *qkv_w = nullptr, *proj_w = nullptr, *fc1_w = nullptr, *fc2_w = nullptr;
        const void *qkv_b = nullptr, *proj_b = nullptr, *fc1_b = nullptr, *fc2_b = nullptr;  // activation dtype
        const float *norm1_g = nullptr, *norm1_b = nullptr, *norm2_g = nullptr, *norm2_b = nullptr;
    };
    float *upload_wide(const std::string &name, int64_t n);   // BF16 file tensor -> device FP32, kept alive
    void *upload_wide_bits(const uint16_t *bits, int64_t n);  // device bits -> device FP32 widening, kept alive
    std::vector<float> read_back_wide(const uint16_t *dev_bits, int64_t n, const char *fn);  // D2H + widen

    std::string path_;
    std::unique_ptr<StrixwDevice> w_;
    kernels::Act act_;
    VisionTowerDims dims_;
    std::vector<int64_t> patch_shape_;  // the file's (rank-5) patch_embed.proj.weight shape
    const uint16_t *patch_w_ = nullptr, *patch_b_bits_ = nullptr, *pos_table_ = nullptr;
    const void *patch_b_ = nullptr;  // patch_b_bits_ or its FP32 widening, by act_
    std::vector<PerBlock> blocks_;
    std::vector<DeviceBuffer<float>> widened_;  // norm gains / biases + F32-run bias widenings, kept alive
    const uint16_t *m_fc1_w_ = nullptr, *m_fc2_w_ = nullptr;
    const void *m_fc1_b_ = nullptr, *m_fc2_b_ = nullptr;  // by act_: raw bits or widened
    const float *m_norm_g_ = nullptr, *m_norm_b_ = nullptr;
};

}  // namespace strix
