#pragma once

// Host image preprocessing for the vision pipeline: the producer that sits between request bytes and the
// engine's consumer, Qwen4ExpSession::forward(..., const ImageInput*, ...). One path: PNG / JPEG bytes ->
// decode (vendored stb, third_party/stb/) -> smart-resize (the processor's rule, with the pixel ceiling
// fixed by the tower's attention cap) -> normalize (mean / std per channel from the checkpoint's
// preprocessor config) -> pack to pixel_values rows (merge-block-major, the frame duplicated across the
// temporal planes) - plus the per-image (t, h, w) position triples and the position step the walk owes
// the next span.
//
// Nothing here is HIP or JSON: the config arrives as values the caller read from the checkpoint's own
// files (nothing is hardcoded blind - the defaults below are this checkpoint's values).
// Every function validates its inputs (STRIX_CHECK, expected vs found); a refusal - undecodable bytes, a
// non-PNG/JPEG format, a non-multiple grid, the aspect-ratio or patch caps - is an exception naming the
// parameter and its value, per the no-fallback rule.

#include <cstdint>
#include <string>
#include <vector>

namespace strix::vision {

// The tower's geometry + the processor's normalization and size bounds. Field order mirrors the
// checkpoint's files (vision_config / preprocessor_config.json); this module is JSON-free - the config
// loader fills these from the checkpoint's own files (the defaults here are this checkpoint's).
// The resize filter preprocess_image uses. BicubicAntialias (the default) is the processor's own filter family
// (cubic a = -0.5 widened by the downscale factor); Bilinear (no antialias) stays for comparison.
enum class ResizeFilter { Bilinear, BicubicAntialias };

struct PreprocConfig {
    int64_t patch_size = 16;            // vision_config.patch_size
    int64_t temporal_patch_size = 2;    // vision_config.temporal_patch_size (the frame duplication)
    int64_t spatial_merge_size = 2;     // vision_config.spatial_merge_size
    int64_t in_channels = 3;            // vision_config.in_channels
    float mean[3] = {0.5f, 0.5f, 0.5f};  // preprocessor_config.image_mean (per channel)
    float std[3] = {0.5f, 0.5f, 0.5f};   // preprocessor_config.image_std (per channel)
    int64_t min_pixels = 65536;         // preprocessor_config.size.shortest_edge
    int64_t max_pixels = 16777216;      // preprocessor_config.size.longest_edge (clamped per max_patches below)
    // The attention cap (kernels/vision.hpp vision_attn, 8192 patches): when > 0 it clamps the resize's
    // max_pixels so the patch grid satisfies grid_h * grid_w <= max_patches: an oversized image is served
    // downscaled, never refused by the kernel. <= 0 = no clamp (a tiled kernel would say so here).
    int64_t max_patches = 8192;
    int64_t max_side = 65536;           // decode refuse: stb takes int dimensions; guard at their range
    int64_t max_decode_bytes = 512LL << 20;  // decoded RGB byte cap (a refused image costs nothing)
    ResizeFilter resize = ResizeFilter::BicubicAntialias;  // the resize filter (see ResizeFilter)

    // The multiple both resize sides round to (the processor's factor: patch x merge = 32).
    int64_t factor() const { return patch_size * spatial_merge_size; }
    // The resize's effective pixel ceiling: max_pixels clamped so the patch grid fits the cap
    // (patch^2 * max_patches). Throws on an inconsistent config (named field by field).
    int64_t effective_max_pixels() const;
    // Validates every field (positive sizes, merge >= 1, finite nonzero std, factors that divide the cap
    // cleanly, min_pixels <= effective max). Throws naming the field and its value.
    void validate() const;
};

// The resize rule the processor applies (smart_resize): round both sides
// to a multiple of cfg.factor(), then scale down / up so the pixel count lands in
// [min_pixels, effective_max_pixels()], keeping the aspect. Refusals: a non-positive side, an aspect ratio
// over 200, an upscale target past max_side. Returns the target (height, width) and the patch grid
// (height / patch, width / patch) - the grid axes grid_thw carries (t = 1 for an image).
struct ResizeTarget {
    int64_t height = 0, width = 0;   // the resize target, both multiples of cfg.factor()
    int64_t grid_h = 0, grid_w = 0;  // height / patch_size, width / patch_size
};
ResizeTarget smart_resize(int64_t height, int64_t width, const PreprocConfig &cfg);

// A decoded, still-quantized image: uint8 RGB, row-major, no stride padding, bottom-up-free (stb's
// top-down default). Float image content is not representable in 8 bits - a file round-trip moves every
// pixel by up to half a quantization step - which is why preprocess_scene takes the float form directly.
struct ImageU8 {
    int64_t width = 0, height = 0;
    std::string format;              // what the bytes were: "png", "jpeg", "gif" (a refusal or an accepted decode)
    std::vector<uint8_t> rgb;        // width * height * 3
};

// The formats the preprocessing accepts from the wire (image parts arrive as data URLs). Bytes
// whose magic names something else refuse, naming the observed magic in hex.
bool is_supported_image_format(const uint8_t *bytes, size_t size);

// Decode PNG / JPEG / GIF (first frame) / BMP / TGA (stb) or WebP (the vendored libwebp) bytes to 8-bit RGB; grey
// and alpha convert as the processor does (grey replicated, alpha dropped). Refusals (expected vs found, always):
// fewer than the magic bytes needed, any other format, a channel count outside 1..4, a
// dimension past cfg.max_side, decoded pixels past cfg.max_decode_bytes, an out-of-memory from the
// decoder, and any stb parse failure (its failure reason quoted).
ImageU8 decode_image(const uint8_t *bytes, size_t size, const PreprocConfig &cfg);
inline ImageU8 decode_image(const std::vector<uint8_t> &bytes, const PreprocConfig &cfg) {
    return decode_image(bytes.data(), bytes.size(), cfg);
}

// Bilinear resize of an 8-bit image to an exact target size: the source pixel centers sit at
// src = (dst + 0.5) * src / dst - 0.5 (PyTorch's half-pixel centers, align_corners=False).
// Separable (horizontal then vertical), weights summed with a 1e-3 absolute guard on top of
// the relative tolerance, clamped to the border pixel. The target must be >= 1 and its pixel count must
// fit size_t; a 1:1 copy when the sizes already match.
ImageU8 resize_bilinear_u8(const ImageU8 &src, int64_t target_height, int64_t target_width);

// Bicubic resize with antialias: per output pixel a cubic (a = -0.5) window centered at (dst + 0.5) * scale,
// stretched by max(scale, 1) when downscaling so every source pixel under it contributes, cut at the image edge and
// renormalized. Separable, horizontal then vertical, each pass rounded to 8 bits. Same target checks as bilinear.
ImageU8 resize_bicubic_aa_u8(const ImageU8 &src, int64_t target_height, int64_t target_width);

// What the engine's ImageInput consumes for one image, produced from a decode or an FP32 scene:
struct Preprocessed {
    std::vector<float> pixel_values;  // [grid_h * grid_w, in_channels x temporal x patch x patch] FP32,
                                      // rows merge-block-major (block row, block col, in-block row, in-block
                                      // col), within a row (channel, temporal, row, col) with the single
                                      // frame duplicated across the temporal planes
    int64_t grid_t = 1, grid_h = 0, grid_w = 0;
    int64_t resized_height = 0, resized_width = 0;    // what was packed == what the grid says
    int64_t merged_tokens = 0;  // grid_h * grid_w / merge^2: the image_pad expansion count, the forward's row count
};

// Decode -> resize (to smart_resize's target) -> normalize ((v / 255 - mean) / std per channel) -> pack.
// cfg's merge grid rule fires here: a target side not a multiple of patch * merge refuses (it cannot
// happen from smart_resize; from a hand-built cfg it must not silently mis-pack).
Preprocessed preprocess_image(const uint8_t *bytes, size_t size, const PreprocConfig &cfg);
inline Preprocessed preprocess_image(const std::vector<uint8_t> &bytes, const PreprocConfig &cfg) {
    return preprocess_image(bytes.data(), bytes.size(), cfg);
}
Preprocessed preprocess_image(const ImageU8 &decoded, const PreprocConfig &cfg);

// The same normalization + packing from FP32 pixels in [0, 1] (3 x height x width, row-major planes,
// already at their final size): the form the reference processor works with - its scenes are float,
// rendered at the resize target, so this path enters after the resize.
// Refusals: null / empty / size-mismatched pixels, non-positive sides, a side not a multiple of patch,
// a merge-multiple violation, a grid past the patch cap (cfg.max_patches).
Preprocessed preprocess_scene(const float *rgb_planar01, int64_t height, int64_t width, const PreprocConfig &cfg);

// The per-image (t, h, w) position triples of one image block (the get_rope_index rule): rows in the
// merger's order (merged cells block-major), each row
// t = start, h = start + cell_row, w = start + cell_col; the walk's next span starts at
// start + max(grid_h, grid_w) / 2 (the block's max position + 1). Refusals: negative start, grid axes
// under 2 or not multiples of merge, a block whose highest position reaches max_positions (when > 0).
struct ImageBlockPositions {
    std::vector<int64_t> pos3;            // 3 * merged_tokens host int64, section positions (not row indices)
    int64_t next_start = 0;               // the walk's current_pos after the block
    int64_t merged_tokens = 0;            // (grid_h / merge) * (grid_w / merge) - the image_token_id copies
};
ImageBlockPositions image_block_positions(int64_t start, int64_t grid_h, int64_t grid_w,
                                          const PreprocConfig &cfg, int64_t max_positions = 0);
// The same from a processed image (its grid + merge).
ImageBlockPositions image_block_positions(int64_t start, const Preprocessed &img, const PreprocConfig &cfg,
                                          int64_t max_positions = 0);

}  // namespace strix::vision
