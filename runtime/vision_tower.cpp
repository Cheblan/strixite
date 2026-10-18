#include "runtime/vision_tower.hpp"

#include "common/bf16.hpp"
#include "common/check.hpp"
#include "kernels/residual.hpp"
#include "kernels/vision.hpp"

#include <cmath>
#include <cstring>
#include <set>

namespace strix {

namespace {

const std::string kV = "model.visual.";
constexpr float kLayernormEps = 1e-6f;    // vision_config layer_norm_eps (the trunk's RMSNorms use the same)
constexpr double kVisionRopeTheta = 1e4;  // vision_config rope_theta (the trunk's is 1e7)
// Heads: the shapes can't split hidden into heads x head_dim, so heads is the architecture's (config.json
// vision_config.num_heads) with the head_dim's rope legality cross-checked - the same convention as the
// hardcoded trunk dimensions.
constexpr int64_t kVisionHeads = 16;
constexpr int64_t kElemsCap = 1ll << 40;  // same element-count ceiling the kernel launchers use

std::string shape_str_local(const std::vector<int64_t> &v) {
    std::string out = "[";
    for (size_t i = 0; i < v.size(); ++i) out += (i ? " x " : "") + std::to_string(v[i]);
    return out + "]";
}

std::vector<float> widen_f32(const uint16_t *bits, int64_t n) {
    std::vector<float> out((size_t)n);
    for (int64_t i = 0; i < n; ++i) out[(size_t)i] = bf16_bits_to_f32(bits[i]);
    return out;
}

}  // namespace

VisionTower::VisionTower(const std::string &path, const StrixwFile &text_weights, kernels::Act act)
    : path_(path), act_(act) {
    const char *fn = "VisionTower";
    STRIX_CHECK(act == kernels::Act::F32 || act == kernels::Act::BF16, fn, ": unsupported activation dtype ",
                (int)act);
    // Headers first (a cheap mmap): refuse a file that isn't this tower before its 856 MiB go to the
    // device - a text weights file named as the vision file fails here, not as a half-loaded model.
    const StrixwFile index(path);
    STRIX_CHECK(index.meta("arch") == "qwen4exp" && index.meta("kind") == "vision", fn, ": '", path,
                "' is arch '", index.meta("arch"), "', kind '", index.meta("kind"), "' - expected a qwen4exp",
                " vision conversion (tools/convert_qwen4exp --vision)");
    STRIX_CHECK(index.meta("layout") == "bf16", fn, ": '", path, "' has layout '", index.meta("layout"),
                "' - the tower is BF16 exact, nothing quantized");
    // The side-file rule the n-gram table follows: converted from the same checkpoint as the text weights.
    const std::string &want = text_weights.meta("source_fingerprint"), &got = index.meta("source_fingerprint");
    STRIX_CHECK(got == want, fn, ": vision weights '", path, "' were converted from checkpoint ", got,
                ", the text weights from ", want, " - convert both from the same checkpoint");
    STRIX_CHECK(!index.meta("vision_config").empty(), fn, ": '", path, "' carries no vision_config metadata");
    // Always the plain Device placement, whatever the text model asked for: HostExperts is a loader
    // contract about the expert stacks (it refuses a file that has none), and the tower has no experts -
    // 856 MiB of BF16 is a device region on every machine.
    w_ = std::make_unique<StrixwDevice>(path, 8, WeightPlacement::Device);
    const StrixwFile &f = w_->file();

    // Dimensions from the tensor shapes: hidden and the pixel width from the patch embed, the MLP width
    // and the merge from the block / merger matrices, the position-table side from its row count.
    const auto shape_of = [&](const std::string &name) -> const std::vector<int64_t> & {
        const StrixwTensor *t = f.find(name);
        STRIX_CHECK(t != nullptr, fn, ": '", path, "' has no tensor '", name, "'");
        STRIX_CHECK(t->encoding == StrixwEncoding::BF16, fn, ": '", name, "' encoding ",
                    strixw_encoding_name(t->encoding), ", expected bf16");
        return t->shape;
    };
    const std::vector<int64_t> &pe = shape_of(kV + "patch_embed.proj.weight");
    // The Conv3d weight keeps the checkpoint's rank 5 [hidden, in_channels, temporal, patch, patch]; the
    // tower GEMM sees its trailing dimensions as one packed pixel row (1536 columns).
    STRIX_CHECK(pe.size() == 2 || pe.size() == 5, fn, ": patch_embed.proj.weight is rank ", pe.size(),
                ", expected [hidden, in x temporal x patch x patch] (rank 2 or 5)");
    int64_t patch_in = 1;
    for (size_t i = 1; i < pe.size(); ++i) {
        STRIX_CHECK(pe[i] >= 1 && patch_in <= kElemsCap / pe[i], fn, ": patch_embed shape dimension ", i,
                    " = ", pe[i], " in ", shape_str_local(pe));
        patch_in *= pe[i];
    }
    STRIX_CHECK(pe.size() >= 1 && pe[0] >= 1, fn, ": patch_embed.proj.weight rank ", pe.size(), " with a "
                "zero-length dimension");
    dims_.hidden = pe[0], dims_.patch_in = patch_in;
    patch_shape_ = pe;
    const std::vector<int64_t> &qkv0 = shape_of(kV + "blocks.0.attn.qkv.weight");
    STRIX_CHECK(qkv0.size() == 2 && qkv0[0] == 3 * dims_.hidden && qkv0[1] == dims_.hidden, fn,
                ": blocks.0.attn.qkv.weight is [", qkv0[0], ", ", qkv0[1], "], expected [", 3 * dims_.hidden,
                ", ", dims_.hidden, "]");
    dims_.heads = kVisionHeads;
    STRIX_CHECK(dims_.hidden % dims_.heads == 0 && (dims_.hidden / dims_.heads) % 4 == 0, fn, ": hidden ",
                dims_.hidden, " doesn't split into ", dims_.heads, " heads with a multiple-of-4 head_dim",
                " (axial rope needs an even half of even count)");
    const std::vector<int64_t> &fc1 = shape_of(kV + "blocks.0.mlp.linear_fc1.weight");
    STRIX_CHECK(fc1.size() == 2 && fc1[1] == dims_.hidden && fc1[0] >= 1, fn, ": mlp.linear_fc1.weight is [",
                fc1[0], ", ", fc1[1], "], expected [inter, ", dims_.hidden, "]");
    dims_.inter = fc1[0];
    const std::vector<int64_t> &mfc2 = shape_of(kV + "merger.linear_fc2.weight");
    STRIX_CHECK(mfc2.size() == 2 && mfc2[0] >= 1 && mfc2[1] >= dims_.hidden && mfc2[1] % dims_.hidden == 0, fn,
                ": merger.linear_fc2.weight is [", mfc2[0], ", ", mfc2[1], "], expected [out, m^2 * ",
                dims_.hidden, "]");
    dims_.merge = (int64_t)std::llround(std::sqrt((double)(mfc2[1] / dims_.hidden)));
    STRIX_CHECK(dims_.merge >= 1 && dims_.merge * dims_.merge * dims_.hidden == mfc2[1], fn, ": merger input "
                "width ", mfc2[1], " is not a square multiple of hidden ", dims_.hidden);
    dims_.out_hidden = mfc2[0];
    const std::vector<int64_t> &pt = shape_of(kV + "pos_embed.weight");
    STRIX_CHECK(pt.size() == 2 && pt[1] == dims_.hidden && pt[0] >= 1, fn, ": pos_embed.weight is [", pt[0],
                ", ", pt[1], "], expected [side^2, ", dims_.hidden, "]");
    dims_.pos_side = (int64_t)std::llround(std::sqrt((double)pt[0]));
    STRIX_CHECK(dims_.pos_side * dims_.pos_side == pt[0], fn, ": the position table has ", pt[0],
                " rows, not a square side");

    patch_w_ = w_->bf16(kV + "patch_embed.proj.weight", patch_shape_);
    pos_table_ = w_->bf16(kV + "pos_embed.weight", {dims_.pos_side * dims_.pos_side, dims_.hidden});
    patch_b_bits_ = w_->bf16(kV + "patch_embed.proj.bias", {dims_.hidden});
    patch_b_ = act_ == kernels::Act::BF16 ? (const void *)patch_b_bits_
                                           : (const void *)upload_wide(kV + "patch_embed.proj.bias",
                                                                       dims_.hidden);

    while (f.find(kV + "blocks." + std::to_string(dims_.depth) + ".attn.qkv.weight") != nullptr) {
        const std::string P = kV + "blocks." + std::to_string(dims_.depth) + ".";
        const int64_t hh = dims_.hidden;
        PerBlock pb;
        pb.qkv_w = w_->bf16(P + "attn.qkv.weight", {3 * hh, hh});
        pb.proj_w = w_->bf16(P + "attn.proj.weight", {hh, hh});
        pb.fc1_w = w_->bf16(P + "mlp.linear_fc1.weight", {dims_.inter, hh});
        pb.fc2_w = w_->bf16(P + "mlp.linear_fc2.weight", {hh, dims_.inter});
        const uint16_t *qkv_b = w_->bf16(P + "attn.qkv.bias", {3 * hh});
        const uint16_t *proj_b = w_->bf16(P + "attn.proj.bias", {hh});
        const uint16_t *f1b = w_->bf16(P + "mlp.linear_fc1.bias", {dims_.inter});
        const uint16_t *f2b = w_->bf16(P + "mlp.linear_fc2.bias", {hh});
        pb.norm1_g = upload_wide(P + "norm1.weight", hh);
        pb.norm1_b = upload_wide(P + "norm1.bias", hh);
        pb.norm2_g = upload_wide(P + "norm2.weight", hh);
        pb.norm2_b = upload_wide(P + "norm2.bias", hh);
        if (act_ == kernels::Act::BF16) {  // biases live in the activation dtype (the kernels' contract)
            pb.qkv_b = qkv_b, pb.proj_b = proj_b, pb.fc1_b = f1b, pb.fc2_b = f2b;
        } else {
            pb.qkv_b = upload_wide_bits(qkv_b, 3 * hh);
            pb.proj_b = upload_wide_bits(proj_b, hh);
            pb.fc1_b = upload_wide_bits(f1b, dims_.inter);
            pb.fc2_b = upload_wide_bits(f2b, hh);
        }
        blocks_.push_back(pb);
        ++dims_.depth;
    }
    STRIX_CHECK(dims_.depth >= 1, fn, ": '", path, "' has no model.visual.blocks.* tensors");

    const int64_t mw = dims_.merge * dims_.merge * dims_.hidden;
    m_fc1_w_ = w_->bf16(kV + "merger.linear_fc1.weight", {mw, mw});
    m_fc2_w_ = w_->bf16(kV + "merger.linear_fc2.weight", {dims_.out_hidden, mw});
    const uint16_t *mfc1_b = w_->bf16(kV + "merger.linear_fc1.bias", {mw});
    const uint16_t *mfc2_b = w_->bf16(kV + "merger.linear_fc2.bias", {dims_.out_hidden});
    if (act_ == kernels::Act::BF16) {
        m_fc1_b_ = mfc1_b, m_fc2_b_ = mfc2_b;
    } else {
        m_fc1_b_ = upload_wide_bits(mfc1_b, mw), m_fc2_b_ = upload_wide_bits(mfc2_b, dims_.out_hidden);
    }
    m_norm_g_ = upload_wide(kV + "merger.norm.weight", dims_.hidden);
    m_norm_b_ = upload_wide(kV + "merger.norm.bias", dims_.hidden);

    // Nothing unexpected: every tensor of the file must be one this code reads (an unaccounted tensor is
    // a spec gap and fails at load - the same rule the converter applies when writing the file).
    std::set<std::string> known;
    for (const char *s : {"patch_embed.proj.weight", "patch_embed.proj.bias", "pos_embed.weight",
                          "merger.norm.weight", "merger.norm.bias", "merger.linear_fc1.weight",
                          "merger.linear_fc1.bias", "merger.linear_fc2.weight", "merger.linear_fc2.bias"})
        known.insert(std::string(kV) + s);
    for (int64_t b = 0; b < dims_.depth; ++b)
        for (const char *s : {"attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight", "attn.proj.bias",
                              "mlp.linear_fc1.weight", "mlp.linear_fc1.bias", "mlp.linear_fc2.weight",
                              "mlp.linear_fc2.bias", "norm1.weight", "norm1.bias", "norm2.weight",
                              "norm2.bias"})
            known.insert(kV + "blocks." + std::to_string(b) + "." + s);
    for (const StrixwTensor &t : f.tensors())
        STRIX_CHECK(known.count(t.name) != 0, fn, ": '", path, "' carries '", t.name,
                    "', which this tower code doesn't read - refusing a file with unaccounted tensors");
}

std::vector<float> VisionTower::read_back_wide(const uint16_t *dev_bits, int64_t n, const char *fn) {
    std::vector<uint16_t> host((size_t)n);
    STRIX_HIP_CHECK(hipMemcpy(host.data(), dev_bits, (size_t)n * 2, hipMemcpyDeviceToHost), fn,
                    ": reading ", n, " BF16 values back to widen them");
    return widen_f32(host.data(), n);
}

float *VisionTower::upload_wide(const std::string &name, int64_t n) {
    const char *fn = "VisionTower::upload_wide";
    const uint16_t *bits = w_->bf16(name, {n});  // device bytes - widened on the host via a load-time copy
    DeviceBuffer<float> buf = DeviceBuffer<float>::from_host(read_back_wide(bits, n, fn), name);
    float *p = buf.get();
    widened_.push_back(std::move(buf));  // kept alive for the tower's whole life
    return p;
}

void *VisionTower::upload_wide_bits(const uint16_t *bits, int64_t n) {
    const char *fn = "VisionTower::upload_wide_bits";
    DeviceBuffer<float> buf = DeviceBuffer<float>::from_host(read_back_wide(bits, n, fn), "vision f32 bias");
    void *p = buf.get();
    widened_.push_back(std::move(buf));
    return p;
}

void VisionTower::grid_taps(int64_t grid_h, int64_t grid_w, std::vector<int32_t> &idx,
                            std::vector<float> &w) const {
    const char *fn = "VisionTower::grid_taps";
    const int64_t m = dims_.merge, side = dims_.pos_side;
    STRIX_CHECK(grid_h >= 1 && grid_w >= 1, fn, ": grid ", grid_h, "x", grid_w);
    STRIX_CHECK(grid_h % m == 0 && grid_w % m == 0, fn, ": grid ", grid_h, "x", grid_w, " not divisible by "
                "the merge side ", m);
    const int64_t n = grid_h * grid_w;
    // Per-axis align_corners bilinear: src = i * (side - 1) / (size - 1) (0 when size == 1), tap indices
    // clamped to the table border, weights (1 - frac, frac); rows in merge-block order like the patches.
    // The arithmetic is FP64 - the exact value of the formula. The reference run records its own
    // FP32-rounded coordinates, so the last bits differ by under 5e-6 on every reference grid the tower
    // is checked against (the bound the engine's tap comparison carries).
    auto axis = [&](int64_t size, std::vector<int64_t> &src, std::vector<float> &wgt) {
        src.assign((size_t)(size * 2), 0);
        wgt.assign((size_t)(size * 2), 0.f);
        for (int64_t i = 0; i < size; ++i) {
            const double s = size == 1 ? 0.0 : (double)i * (double)(side - 1) / (double)(size - 1);
            const int64_t fl = (int64_t)std::floor(s);
            const float frac = (float)(s - (double)fl);
            src[(size_t)(i * 2)] = std::min(side - 1, std::max<int64_t>(0, fl));
            src[(size_t)(i * 2 + 1)] = std::min(side - 1, fl + 1);
            wgt[(size_t)(i * 2)] = 1.f - frac;
            wgt[(size_t)(i * 2 + 1)] = frac;
        }
    };
    std::vector<int64_t> h_src, w_src;
    std::vector<float> h_wgt, w_wgt;
    axis(grid_h, h_src, h_wgt);
    axis(grid_w, w_src, w_wgt);
    idx.assign((size_t)(n * 4), 0);
    w.assign((size_t)(n * 4), 0.f);
    const int64_t blocks_w = grid_w / m, m2 = m * m;
    for (int64_t p = 0; p < n; ++p) {
        const int64_t blk = p / m2, sub = p % m2;
        const int64_t row = (blk / blocks_w) * m + sub / m, col = (blk % blocks_w) * m + sub % m;
        for (int a = 0; a < 2; ++a)
            for (int c = 0; c < 2; ++c) {
                idx[(size_t)(p * 4 + a * 2 + c)] =
                    (int32_t)(h_src[(size_t)(row * 2 + a)] * side + w_src[(size_t)(col * 2 + c)]);
                w[(size_t)(p * 4 + a * 2 + c)] = h_wgt[(size_t)(row * 2 + a)] * w_wgt[(size_t)(col * 2 + c)];
            }
    }
}

TowerScratchBytes VisionTower::scratch_bytes(int64_t grid_h, int64_t grid_w) const {
    const int64_t n = grid_h * grid_w, m2 = dims_.merge * dims_.merge;
    TowerScratchBytes b;
    b.patch_in = n * dims_.patch_in;
    b.hidden = n * dims_.hidden;
    b.qkv = n * 3 * dims_.hidden;
    b.mlp = n * dims_.inter;
    b.merg = (n / m2) * m2 * dims_.hidden;
    b.merged = (n / m2) * dims_.out_hidden;
    return b;
}

void VisionTower::forward(int64_t grid_h, int64_t grid_w, const int32_t *idx_dev, const float *w_dev,
                          const TowerScratch &s, hipStream_t stream) const {
    const char *fn = "VisionTower::forward";
    const int64_t m = dims_.merge, h = dims_.hidden, m2 = m * m, n = grid_h * grid_w;
    STRIX_CHECK(grid_h >= m && grid_w >= m && grid_h % m == 0 && grid_w % m == 0, fn, ": grid ", grid_h, "x",
                grid_w, ", expected positive multiples of the merge side ", m);
    STRIX_CHECK(n <= kernels::kMaxVisionSeq, fn, ": ", n, " patches exceeds the dense-attention cap ",
                kernels::kMaxVisionSeq);
    STRIX_CHECK(idx_dev != nullptr && w_dev != nullptr, fn, ": null position-tap arrays");
    const void *bufs[] = {s.pixel_values, s.x, s.conv, s.ln, s.proj, s.qkv, s.mlp, s.merg, s.merged};
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); ++i)
        STRIX_CHECK(bufs[i] != nullptr, fn, ": scratch buffer #", i, " is null (see TowerScratch)");

    // The launch sequence the tower block-level and end-to-end checks gate: patch GEMM (the Conv3d with
    // kernel = stride = (temporal, patch, patch) is a GEMM over the packed pixel rows) + bias, the
    // resampled position table add, then per block LN1 / qkv GEMM + bias / axial rope / dense attention /
    // output GEMM + bias / residual, LN2 / fc1 + bias / tanh GELU / fc2 + bias / residual; the merger's
    // LN, 2x2 regroup, fc1 + bias, erf GELU, fc2 + bias.
    kernels::vision_linear_bias(s.pixel_values, patch_w_, patch_b_, s.x, n, h, dims_.patch_in, act_, stream);
    kernels::vision_pos_add(s.x, pos_table_, idx_dev, w_dev, s.x, n, h, 4, act_, stream);
    for (const PerBlock &pb : blocks_) {
        kernels::vision_layernorm(s.x, pb.norm1_g, pb.norm1_b, s.ln, n, h, kLayernormEps, act_, stream);
        kernels::vision_linear_bias(s.ln, pb.qkv_w, pb.qkv_b, s.qkv, n, 3 * h, h, act_, stream);
        kernels::vision_rope_qk(s.qkv, n, dims_.heads, dims_.head_dim(), grid_h, grid_w, m,
                                (float)kVisionRopeTheta, act_, stream);
        kernels::vision_attn(s.qkv, s.proj, n, dims_.heads, dims_.head_dim(), act_, stream);
        kernels::vision_linear_bias(s.proj, pb.proj_w, pb.proj_b, s.ln, n, h, h, act_, stream);
        kernels::residual_add(s.x, s.ln, n * h, act_, stream);
        kernels::vision_layernorm(s.x, pb.norm2_g, pb.norm2_b, s.ln, n, h, kLayernormEps, act_, stream);
        kernels::vision_linear_bias(s.ln, pb.fc1_w, pb.fc1_b, s.mlp, n, dims_.inter, h, act_, stream);
        kernels::vision_gelu(s.mlp, n * dims_.inter, kernels::VisionGelu::Tanh, act_, stream);
        kernels::vision_linear_bias(s.mlp, pb.fc2_w, pb.fc2_b, s.proj, n, h, dims_.inter, act_, stream);
        kernels::residual_add(s.x, s.proj, n * h, act_, stream);
    }
    kernels::vision_layernorm(s.x, m_norm_g_, m_norm_b_, s.ln, n, h, kLayernormEps, act_, stream);
    kernels::vision_merge_regroup(s.ln, s.merg, n, h, m, act_, stream);
    kernels::vision_linear_bias(s.merg, m_fc1_w_, m_fc1_b_, s.qkv, n / m2, m2 * h, m2 * h, act_, stream);
    kernels::vision_gelu(s.qkv, (n / m2) * m2 * h, kernels::VisionGelu::Erf, act_, stream);
    kernels::vision_linear_bias(s.qkv, m_fc2_w_, m_fc2_b_, s.merged, n / m2, dims_.out_hidden, m2 * h, act_,
                                stream);
}

}  // namespace strix
