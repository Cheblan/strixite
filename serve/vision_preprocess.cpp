// Host image preprocessing for the vision pipeline - see serve/vision_preprocess.hpp for the contract.
// The resize rule mirrors the processor's smart_resize with max_pixels clamped to the attention cap;
// the normalization and packing mirror its rescale + normalize + patchify.

#include "serve/vision_preprocess.hpp"

#include "common/check.hpp"

#include <src/webp/decode.h>  // third_party/libwebp (the vendored decoder)
#include <stb_image.h>  // third_party/stb (declarations; the impl is in third_party/stb_impl)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace strix::vision {

namespace {

constexpr int kStbMagicBytes = 8;

bool starts_with(const uint8_t *bytes, size_t size, const char *magic, size_t magic_size) {
    return size >= magic_size && std::memcmp(bytes, magic, magic_size) == 0;
}

std::string hex_prefix(const uint8_t *bytes, size_t size, size_t want) {
    static const char *digits = "0123456789abcdef";
    const size_t n = size < want ? size : want;
    std::string out;
    for (size_t i = 0; i < n; ++i) {
        out += "0x";
        out += digits[bytes[i] >> 4];
        out += digits[bytes[i] & 0xf];
        if (i + 1 < n) out += ' ';
    }
    return out;
}

}  // namespace

int64_t PreprocConfig::effective_max_pixels() const {
    if (max_patches <= 0) return max_pixels;  // no clamp (a tiled kernel would say so here)
    STRIX_CHECK(patch_size >= 1 && patch_size <= 65536,
                "PreprocConfig.patch_size is ", patch_size, ", expected 1..65536");
    const int64_t per_patch = patch_size * patch_size;  // one patch = patch^2 pixels
    STRIX_CHECK(max_patches <= std::numeric_limits<int64_t>::max() / per_patch,
                "PreprocConfig: max_patches ", max_patches, " x patch^2 ", per_patch, " overflows");
    const int64_t clamped = per_patch * max_patches;  // grid_h * grid_w <= max_patches in pixels
    return std::min(max_pixels, clamped);
}

void PreprocConfig::validate() const {
    STRIX_CHECK(patch_size >= 1, "PreprocConfig.patch_size is ", patch_size, ", expected >= 1");
    STRIX_CHECK(temporal_patch_size >= 1, "PreprocConfig.temporal_patch_size is ", temporal_patch_size,
                ", expected >= 1");
    STRIX_CHECK(spatial_merge_size >= 1, "PreprocConfig.spatial_merge_size is ", spatial_merge_size,
                ", expected >= 1");
    STRIX_CHECK(in_channels >= 1 && in_channels <= 3, "PreprocConfig.in_channels is ", in_channels,
                ", expected 1..3 (this pipeline decodes to 3-channel RGB)");
    STRIX_CHECK(min_pixels >= 1, "PreprocConfig.min_pixels is ", min_pixels, ", expected >= 1");
    STRIX_CHECK(max_pixels >= 1, "PreprocConfig.max_pixels is ", max_pixels, ", expected >= 1");
    STRIX_CHECK(max_side >= 1, "PreprocConfig.max_side is ", max_side, ", expected >= 1 (stb dimensions "
                "are ints; this is their cap)");
    STRIX_CHECK(max_decode_bytes >= 1, "PreprocConfig.max_decode_bytes is ", max_decode_bytes,
                ", expected >= 1");
    for (int c = 0; c < 3; ++c) {
        STRIX_CHECK(std::isfinite(mean[c]), "PreprocConfig.mean[", c, "] is ", mean[c],
                    ", expected finite");
        STRIX_CHECK(std::isfinite(std[c]) && std[c] != 0.0f, "PreprocConfig.std[", c, "] is ", std[c],
                    ", expected finite and nonzero");
    }
    if (max_patches > 0) {  // the cap must divide into the factor grid, or its clamp is a lie
        STRIX_CHECK(max_patches % 4 == 0, "PreprocConfig.max_patches is ", max_patches,
                    ", expected a multiple of 4 (2 x 2 merge: merged rows are max_patches / 4)");
    }
    const int64_t factor = this->factor();
    STRIX_CHECK(factor <= (1 << 24), "PreprocConfig: patch_size ", patch_size, " x spatial_merge_size ",
                spatial_merge_size, " (factor ", factor, ") exceeds 16777216 - the resize's rounded "
                "targets would not stay within int64 pixel products");
    effective_max_pixels();  // range / overflow checks for max_patches x patch^2
    STRIX_CHECK(min_pixels <= effective_max_pixels(), "PreprocConfig.min_pixels is ", min_pixels, " > the "
                "effective max_pixels ", effective_max_pixels(), " (max_pixels ", max_pixels, ", max_patches ",
                max_patches, ") - no target would satisfy the rule");
}

ResizeTarget smart_resize(int64_t height, int64_t width, const PreprocConfig &cfg) {
    cfg.validate();
    STRIX_CHECK(height > 0 && width > 0, "smart_resize: height ", height, ", width ", width,
                " - both dimensions must be >= 1");
    const int64_t longer = std::max(height, width), shorter = std::min(height, width);
    const double aspect = (double)longer / (double)shorter;
    STRIX_CHECK(!(aspect > 200.0), "smart_resize: aspect ratio ", aspect, " (", height, "x", width,
                ") exceeds the processor's 200 limit");
    const int64_t factor = cfg.factor();
    STRIX_CHECK(height <= cfg.max_side && width <= cfg.max_side, "smart_resize: ", height, "x", width,
                " has a side past the config's max_side ", cfg.max_side);
    // Python's round() = nearest, ties to even: nearbyint with the default rounding mode.
    int64_t h_bar = (int64_t)std::nearbyint((double)height / (double)factor) * factor;
    int64_t w_bar = (int64_t)std::nearbyint((double)width / (double)factor) * factor;
    const int64_t max_px = cfg.effective_max_pixels();
    if (h_bar * w_bar > max_px) {
        const double beta = std::sqrt(((double)height * (double)width) / (double)max_px);
        h_bar = std::max(factor, (int64_t)std::floor(((double)height / beta) / (double)factor) * factor);
        w_bar = std::max(factor, (int64_t)std::floor(((double)width / beta) / (double)factor) * factor);
    } else if (h_bar * w_bar < cfg.min_pixels) {
        const double beta = std::sqrt((double)cfg.min_pixels / ((double)height * (double)width));
        h_bar = (int64_t)std::ceil(((double)height * beta) / (double)factor) * factor;
        w_bar = (int64_t)std::ceil(((double)width * beta) / (double)factor) * factor;
        STRIX_CHECK(h_bar <= cfg.max_side && w_bar <= cfg.max_side, "smart_resize: upscaling ", height,
                    "x", width, " to its min_pixels target gives ", w_bar, "x", h_bar, ", past the config's "
                    "max_side ", cfg.max_side);
    }
    STRIX_CHECK(h_bar >= factor && w_bar >= factor && h_bar % factor == 0 && w_bar % factor == 0,
                "smart_resize: target ", h_bar, "x", w_bar, " is not a positive multiple of the factor ",
                factor, " (from ", height, "x", width, ") - the rule cannot produce this");
    ResizeTarget out;
    out.height = h_bar, out.width = w_bar;
    out.grid_h = h_bar / cfg.patch_size, out.grid_w = w_bar / cfg.patch_size;
    STRIX_CHECK(out.grid_h >= 2 && out.grid_w >= 2 && out.grid_h % cfg.spatial_merge_size == 0 &&
                    out.grid_w % cfg.spatial_merge_size == 0,
                "smart_resize target ", h_bar, "x", w_bar, " (from ", height, "x", width, ") gives patch "
                "grid ", out.grid_h, "x", out.grid_w, " - expected sides of at least 2 patches and a "
                "multiple of the merge ", cfg.spatial_merge_size);
    STRIX_CHECK(cfg.max_patches <= 0 || out.grid_h * out.grid_w <= cfg.max_patches,
                "smart_resize target ", h_bar, "x", w_bar, " gives ", out.grid_h * out.grid_w,
                " patches, past the cap of ", cfg.max_patches);
    return out;
}

namespace {
// TGA has no magic number: a strict check of its 18-byte header instead (color-map type 0 / 1, image type one of the
// six stb decodes, a color-map spec of zeros when there is no color map, nonzero sides, a real pixel depth).
bool looks_like_tga(const uint8_t *b, size_t size) {
    if (size < 18) return false;
    const int cmap = b[1], type = b[2], bpp = b[16];
    if (cmap > 1) return false;
    if (type != 1 && type != 2 && type != 3 && type != 9 && type != 10 && type != 11) return false;
    if ((type == 1 || type == 9) != (cmap == 1)) return false;
    if (cmap == 0 && (b[3] | b[4] | b[5] | b[6] | b[7]) != 0) return false;
    if ((b[12] | (b[13] << 8)) == 0 || (b[14] | (b[15] << 8)) == 0) return false;
    return bpp == 8 || bpp == 15 || bpp == 16 || bpp == 24 || bpp == 32;
}
// The served formats by their leading bytes; "" for none. WebP is RIFF....WEBP.
std::string sniff_format(const uint8_t *bytes, size_t size) {
    static const uint8_t kPng[] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    if (starts_with(bytes, size, reinterpret_cast<const char *>(kPng), sizeof kPng)) return "png";
    if (size >= 3 && bytes[0] == 0xff && bytes[1] == 0xd8 && bytes[2] == 0xff) return "jpeg";
    if (size >= 6 && std::memcmp(bytes, "GIF8", 4) == 0 && (bytes[4] == '7' || bytes[4] == '9') && bytes[5] == 'a')
        return "gif";
    if (size >= 12 && std::memcmp(bytes, "RIFF", 4) == 0 && std::memcmp(bytes + 8, "WEBP", 4) == 0) return "webp";
    if (size >= 2 && bytes[0] == 'B' && bytes[1] == 'M') return "bmp";
    if (looks_like_tga(bytes, size)) return "tga";
    return "";
}
}  // namespace

// WebP through the vendored libwebp decoder: the size caps before any pixel is decoded, then straight into RGB (alpha
// dropped, as the processor's convert("RGB") does; animated files are refused by WebPGetInfo / the decode).
static ImageU8 decode_webp(const uint8_t *bytes, size_t size, const PreprocConfig &cfg) {
    int w = 0, h = 0;
    STRIX_CHECK(WebPGetInfo(bytes, size, &w, &h) != 0, "decode_image: libwebp could not read the webp header of ",
                size, " bytes (corrupt, or an animated / unsupported WebP)");
    STRIX_CHECK(w > 0 && h > 0 && w <= cfg.max_side && h <= cfg.max_side, "decode_image: the webp image is ", w,
                "x", h, ", expected both sides in [1, ", cfg.max_side, "]");
    const int64_t wanted = (int64_t)w * h * 3;
    STRIX_CHECK(wanted <= cfg.max_decode_bytes, "decode_image: ", w, "x", h, " decodes to ", wanted,
                " RGB bytes, past the cap of ", cfg.max_decode_bytes);
    ImageU8 out;
    out.format = "webp";
    out.width = w, out.height = h;
    out.rgb.resize((size_t)wanted);
    STRIX_CHECK(WebPDecodeRGBInto(bytes, size, out.rgb.data(), out.rgb.size(), w * 3) != nullptr,
                "decode_image: libwebp failed to decode the ", w, "x", h, " webp image (", size, " bytes)");
    return out;
}

bool is_supported_image_format(const uint8_t *bytes, size_t size) {
    const std::string f = bytes ? sniff_format(bytes, size) : "";
    return !f.empty();
}

ImageU8 decode_image(const uint8_t *bytes, size_t size, const PreprocConfig &cfg) {
    cfg.validate();
    STRIX_CHECK(bytes != nullptr || size == 0, "decode_image: ", size, " bytes with a null pointer");
    const std::string format = bytes ? sniff_format(bytes, size) : "";
    if (format.empty())
        STRIX_FAIL("decode_image: magic bytes ", hex_prefix(bytes, size, kStbMagicBytes),
                   " - expected PNG (89 50 4E 47 ...), JPEG (FF D8 FF), GIF (GIF87a / GIF89a), BMP (BM), WebP "
                   "(RIFF .... WEBP) or a TGA header");
    if (format == "webp") return decode_webp(bytes, size, cfg);
    STRIX_CHECK(size >= (size_t)kStbMagicBytes, "decode_image: ", size, " bytes - too few to even hold a ",
                format, " header");

    STRIX_CHECK(size <= (size_t)std::numeric_limits<int>::max(), "decode_image: ", size,
                " input bytes, stb takes an int count");
    int w = 0, h = 0, channels_in_file = 0;
    if (!stbi_info_from_memory((const stbi_uc *)bytes, (int)size, &w, &h, &channels_in_file))
        STRIX_FAIL("decode_image: stb could not read the ", format, " header of ", size,
                   " bytes (", stbi_failure_reason(), ")");
    // 1 (grey), 2 (grey + alpha), 3 (RGB), 4 (RGBA) all decode to RGB the way the checkpoint's processor converts
    // them (transformers' convert_to_rgb = PIL convert("RGB")): grey replicated, alpha dropped - not composited.
    // stb's STBI_rgb request does exactly that.
    STRIX_CHECK(channels_in_file >= 1 && channels_in_file <= 4, "decode_image: the ", format, " image has ",
                channels_in_file, " color channels, expected 1..4 (grey, grey + alpha, RGB, RGBA)");
    STRIX_CHECK(w > 0 && h > 0 && w <= cfg.max_side && h <= cfg.max_side, "decode_image: the ", format,
                " image is ", w, "x", h, ", expected both sides in [1, ", cfg.max_side, "]");
    const int64_t wanted = (int64_t)w * h * 3;
    STRIX_CHECK(wanted <= cfg.max_decode_bytes, "decode_image: ", w, "x", h, " decodes to ", wanted,
                " RGB bytes, past the cap of ", cfg.max_decode_bytes);

    ImageU8 out;
    out.format = format;
    out.width = w, out.height = h;
    out.rgb.resize((size_t)wanted);  // stb will not be handed a null buffer (1x1 minimum passes above)
    stbi_uc *got = stbi_load_from_memory((const stbi_uc *)bytes, (int)size, &w, &h, &channels_in_file,
                                         STBI_rgb);
    STRIX_CHECK(got != nullptr, "decode_image: stb failed to decode the ", format, " image (",
                stbi_failure_reason(), ")");
    STRIX_CHECK(w == out.width && h == out.height, "decode_image: stb decoded to ", w, "x", h,
                ", the info pass said ", out.width, "x", out.height);
    std::memcpy(out.rgb.data(), got, (size_t)wanted);
    stbi_image_free(got);
    return out;
}

namespace {
// One output index's taps for the antialiased bicubic: the first source index and its normalized weights.
struct CubicTaps {
    std::vector<int32_t> first, count;
    std::vector<float> w;     // count[d] weights for d, packed at stride `width`
    int64_t width = 0;
};
CubicTaps cubic_aa_axis(int64_t src_n, int64_t dst_n) {
    constexpr double a = -0.5;
    const auto cubic = [](double x) {
        x = std::fabs(x);
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
        if (x < 2.0) return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
        return 0.0;
    };
    const double scale = (double)src_n / (double)dst_n, stretch = std::max(scale, 1.0), support = 2.0 * stretch;
    CubicTaps t;
    t.width = (int64_t)std::ceil(support) * 2 + 1;
    t.first.resize((size_t)dst_n), t.count.resize((size_t)dst_n), t.w.assign((size_t)(dst_n * t.width), 0.f);
    for (int64_t d = 0; d < dst_n; ++d) {
        const double center = ((double)d + 0.5) * scale;
        const int64_t lo = std::max<int64_t>(0, (int64_t)std::floor(center - support + 0.5));
        const int64_t hi = std::min<int64_t>(src_n, (int64_t)std::floor(center + support + 0.5));
        STRIX_CHECK(hi > lo && hi - lo <= t.width, "cubic_aa_axis: window [", lo, ", ", hi, ") for output ", d,
                    " of ", dst_n, " (source ", src_n, "), expected 1..", t.width, " taps");
        double sum = 0;
        std::vector<double> ww((size_t)(hi - lo));
        for (int64_t j = lo; j < hi; ++j) sum += ww[(size_t)(j - lo)] = cubic(((double)j + 0.5 - center) / stretch);
        STRIX_CHECK(sum > 0, "cubic_aa_axis: weight sum ", sum, " for output ", d);
        t.first[(size_t)d] = (int32_t)lo, t.count[(size_t)d] = (int32_t)(hi - lo);
        for (int64_t j = 0; j < hi - lo; ++j) t.w[(size_t)(d * t.width + j)] = (float)(ww[(size_t)j] / sum);
    }
    return t;
}
inline uint8_t round_u8(double v) { return (uint8_t)std::lround(std::min(255.0, std::max(0.0, v))); }
}  // namespace

ImageU8 resize_bicubic_aa_u8(const ImageU8 &src, int64_t target_height, int64_t target_width) {
    STRIX_CHECK(src.width >= 1 && src.height >= 1, "resize_bicubic_aa_u8: source is ", src.width, "x", src.height,
                ", expected both >= 1");
    STRIX_CHECK(src.rgb.size() == (size_t)(src.width * src.height * 3), "resize_bicubic_aa_u8: source has ",
                src.rgb.size(), " bytes, expected ", src.width * src.height * 3);
    STRIX_CHECK(target_height >= 1 && target_width >= 1, "resize_bicubic_aa_u8: target is ", target_height, "x",
                target_width, ", expected both >= 1");
    STRIX_CHECK(target_height <= (int64_t)(std::numeric_limits<int>::max() / 3 / std::max<int64_t>(target_width, 1)),
                "resize_bicubic_aa_u8: target ", target_height, "x", target_width,
                " exceeds the addressable pixel count");
    ImageU8 out;
    out.format = src.format;
    out.width = target_width, out.height = target_height;
    if (target_width == src.width && target_height == src.height) {
        out.rgb = src.rgb;
        return out;
    }
    const CubicTaps h = cubic_aa_axis(src.width, target_width), v = cubic_aa_axis(src.height, target_height);
    std::vector<uint8_t> mid((size_t)(target_width * src.height * 3));
    for (int64_t y = 0; y < src.height; ++y) {
        const uint8_t *row = src.rgb.data() + (size_t)(y * src.width * 3);
        uint8_t *m = mid.data() + (size_t)(y * target_width * 3);
        for (int64_t x = 0; x < target_width; ++x) {
            const float *w = h.w.data() + (size_t)(x * h.width);
            const uint8_t *p = row + (size_t)h.first[(size_t)x] * 3;
            double acc[3] = {0, 0, 0};
            for (int32_t k = 0; k < h.count[(size_t)x]; ++k)
                for (int c = 0; c < 3; ++c) acc[c] += w[k] * p[k * 3 + c];
            for (int c = 0; c < 3; ++c) m[x * 3 + c] = round_u8(acc[c]);
        }
    }
    out.rgb.resize((size_t)(target_width * target_height * 3));
    const int64_t stride = target_width * 3;
    for (int64_t y = 0; y < target_height; ++y) {
        const float *w = v.w.data() + (size_t)(y * v.width);
        const uint8_t *col0 = mid.data() + (size_t)(v.first[(size_t)y] * stride);
        uint8_t *o = out.rgb.data() + (size_t)(y * stride);
        std::vector<double> acc((size_t)stride, 0.0);
        for (int32_t k = 0; k < v.count[(size_t)y]; ++k) {
            const uint8_t *r = col0 + (size_t)(k * stride);
            for (int64_t i = 0; i < stride; ++i) acc[(size_t)i] += w[k] * r[i];
        }
        for (int64_t i = 0; i < stride; ++i) o[i] = round_u8(acc[(size_t)i]);
    }
    return out;
}

ImageU8 resize_bilinear_u8(const ImageU8 &src, int64_t target_height, int64_t target_width) {
    STRIX_CHECK(src.width >= 1 && src.height >= 1, "resize_bilinear_u8: source is ", src.width, "x",
                src.height, ", expected both >= 1");
    STRIX_CHECK(src.rgb.size() == (size_t)(src.width * src.height * 3), "resize_bilinear_u8: source has ",
                src.rgb.size(), " bytes, expected ", src.width * src.height * 3, " for ", src.width, "x",
                src.height);
    STRIX_CHECK(target_height >= 1 && target_width >= 1, "resize_bilinear_u8: target is ", target_height,
                "x", target_width, ", expected both >= 1");
    STRIX_CHECK(target_height <= (int64_t)(std::numeric_limits<int>::max() / 3 / std::max<int64_t>(target_width, 1)),
                "resize_bilinear_u8: target ", target_height, "x", target_width,
                " exceeds the addressable pixel count");
    ImageU8 out;
    out.format = src.format;
    out.width = target_width, out.height = target_height;
    out.rgb.resize((size_t)(target_width * target_height * 3));
    if (target_width == src.width && target_height == src.height) {
        out.rgb = src.rgb;  // the identity case
        return out;
    }

    // Separable: horizontal pass into FP32, then vertical into the uint8 output. Half-pixel centers,
    // border-clamped taps, per-axis weights (a nearest neighbour is the k0 == k1 case).
    const int64_t channels = 3;
    const auto axis = [](int64_t src_n, int64_t dst_n, double scale) {
        struct Per { int32_t k0, k1; float w1; };
        std::vector<Per> out((size_t)dst_n);
        for (int64_t d = 0; d < dst_n; ++d) {
            double src = ((double)d + 0.5) * scale - 0.5;
            if (src < 0.0) src = 0.0;
            const double f0 = std::floor(src);
            int64_t k0 = (int64_t)f0, k1 = k0 + 1;
            if (k0 > src_n - 1) k0 = src_n - 1;
            if (k1 > src_n - 1) k1 = src_n - 1;
            double w1 = src - f0;
            if (w1 < 0.0) w1 = 0.0;
            if (w1 > 1.0) w1 = 1.0;
            out[(size_t)d] = Per{(int32_t)k0, (int32_t)k1, (float)w1};
        }
        return out;
    };
    const auto horiz = axis(src.width, target_width, (double)src.width / (double)target_width);
    const auto vert = axis(src.height, target_height, (double)src.height / (double)target_height);

    std::vector<float> mid((size_t)(target_width * src.height * channels));
    for (int64_t y = 0; y < src.height; ++y) {
        const uint8_t *row = src.rgb.data() + (size_t)(y * src.width * channels);
        float *mrow = mid.data() + (size_t)(y * target_width * channels);
        for (int64_t x = 0; x < target_width; ++x) {
            const auto &t = horiz[(size_t)x];
            for (int64_t c = 0; c < channels; ++c) {
                const double v = row[(size_t)(t.k0 * channels + c)] * (1.0 - (double)t.w1) +
                                 row[(size_t)(t.k1 * channels + c)] * (double)t.w1;
                mrow[(size_t)(x * channels + c)] = (float)v;
            }
        }
    }
    for (int64_t d = 0; d < target_height; ++d) {
        const auto &t = vert[(size_t)d];
        const float *r0 = mid.data() + (size_t)(t.k0 * target_width * channels);
        const float *r1 = mid.data() + (size_t)(t.k1 * target_width * channels);
        uint8_t *orow = out.rgb.data() + (size_t)(d * target_width * channels);
        const double w0 = 1.0 - (double)t.w1, w1 = (double)t.w1;
        for (int64_t x = 0; x < target_width * channels; ++x) {
            const double v = r0[x] * w0 + r1[x] * w1;
            const double rounded = std::nearbyint(v);  // ties to even, like the oracle's .round()
            orow[x] = (uint8_t)(rounded < 0.0 ? 0 : (rounded > 255.0 ? 255 : (int)rounded));
        }
    }
    return out;
}

namespace {

// Shared tail of every preprocessing entry point: the grid checks + the normalize-and-pack loop.
// rgb_planar01 is 3 planes of height x width FP32 values in [0, 1]: the processor's float form - its
// uint8 pixels are put on the same grid before packing (the image path divides by 255.0f there,
// exactly as a u8 / 255.0 division rounds). The pack itself is the normalize ((v - mean) / std per
// channel, stored FP32) in the processor's row layout.
Preprocessed pack_normalized(const float *rgb_planar01, int64_t height, int64_t width,
                             const PreprocConfig &cfg, const char *fn) {
    STRIX_CHECK(rgb_planar01 != nullptr, fn, ": null pixel pointer");
    STRIX_CHECK(height >= 1 && width >= 1, fn, ": ", height, "x", width, " - both must be >= 1");
    STRIX_CHECK(cfg.in_channels == 3, fn, ": in_channels is ", cfg.in_channels,
                ", this pipeline decodes 3-channel RGB");
    STRIX_CHECK(height % cfg.patch_size == 0 && width % cfg.patch_size == 0, fn, ": ", height, "x", width,
                " has sides that are not multiples of patch_size ", cfg.patch_size);
    const int64_t grid_h = height / cfg.patch_size, grid_w = width / cfg.patch_size;
    STRIX_CHECK(grid_h >= cfg.spatial_merge_size && grid_w >= cfg.spatial_merge_size &&
                    grid_h % cfg.spatial_merge_size == 0 && grid_w % cfg.spatial_merge_size == 0,
                fn, ": ", height, "x", width, " gives patch grid ", grid_h, "x", grid_w,
                ", expected both >= ", cfg.spatial_merge_size, " and multiples of the merge ",
                cfg.spatial_merge_size);
    const int64_t patches = grid_h * grid_w;
    STRIX_CHECK(cfg.max_patches <= 0 || patches <= cfg.max_patches, fn, ": grid ", grid_h, "x", grid_w, " is ",
                patches, " patches, past the cap of ", cfg.max_patches);

    Preprocessed out;
    out.grid_t = 1, out.grid_h = grid_h, out.grid_w = grid_w;
    out.resized_height = height, out.resized_width = width;
    out.merged_tokens = patches / (cfg.spatial_merge_size * cfg.spatial_merge_size);
    const int64_t plane = (int64_t)height * width;
    const int64_t pp = cfg.patch_size * cfg.patch_size;
    const int64_t row_len = cfg.in_channels * cfg.temporal_patch_size * pp;
    out.pixel_values.assign((size_t)(patches * row_len), 0.0f);
    const int64_t m = cfg.spatial_merge_size, gw_blocks = grid_w / m;
    for (int64_t b = 0; b < patches; ++b) {
        const int64_t blk = b / (m * m), sub = b % (m * m);
        const int64_t hr = (blk / gw_blocks) * m + sub / m, hc = (blk % gw_blocks) * m + sub % m;
        float *dst = out.pixel_values.data() + (size_t)(b * row_len);
        for (int64_t c = 0; c < cfg.in_channels; ++c) {
            const float *plane_c = rgb_planar01 + c * plane;
            const float value_scale = 1.0f / cfg.std[c];
            const float value_shift = -cfg.mean[c] / cfg.std[c];
            for (int64_t t = 0; t < cfg.temporal_patch_size; ++t) {
                float *tbase = dst + (c * cfg.temporal_patch_size + t) * pp;
                for (int64_t r = 0; r < cfg.patch_size; ++r) {
                    const float *src = plane_c + (hr * cfg.patch_size + r) * width + hc * cfg.patch_size;
                    float *rbase = tbase + r * cfg.patch_size;
                    for (int64_t col = 0; col < cfg.patch_size; ++col)
                        rbase[col] = fmaf(src[col], value_scale, value_shift);  // (v - mean) / std
                }
            }
        }
    }
    return out;
}

}  // namespace

Preprocessed preprocess_scene(const float *rgb_planar01, int64_t height, int64_t width,
                              const PreprocConfig &cfg) {
    cfg.validate();
    return pack_normalized(rgb_planar01, height, width, cfg, "preprocess_scene");
}

Preprocessed preprocess_image(const ImageU8 &decoded, const PreprocConfig &cfg) {
    cfg.validate();
    const ResizeTarget target = smart_resize(decoded.height, decoded.width, cfg);
    const ImageU8 resized = cfg.resize == ResizeFilter::BicubicAntialias
                                ? resize_bicubic_aa_u8(decoded, target.height, target.width)
                                : resize_bilinear_u8(decoded, target.height, target.width);
    std::vector<float> planar((size_t)(3 * resized.height * resized.width));
    const int64_t plane = (int64_t)resized.height * resized.width;
    for (int64_t y = 0; y < resized.height; ++y)
        for (int64_t x = 0; x < resized.width; ++x)
            for (int c = 0; c < 3; ++c)
                planar[(size_t)(c * plane + y * resized.width + x)] =
                    (float)resized.rgb[(size_t)((y * resized.width + x) * 3 + c)] / 255.0f;  // the
                // processor's u8 / 255.0 - a float division (255 is exact), not a reciprocal multiply
    Preprocessed out = pack_normalized(planar.data(), resized.height, resized.width, cfg,
                                       "preprocess_image");
    STRIX_CHECK(out.resized_height == target.height && out.resized_width == target.width,
                "preprocess_image: packed ", out.resized_height, "x", out.resized_width, ", the resize "
                "target said ", target.height, "x", target.width);
    return out;
}

Preprocessed preprocess_image(const uint8_t *bytes, size_t size, const PreprocConfig &cfg) {
    return preprocess_image(decode_image(bytes, size, cfg), cfg);
}

ImageBlockPositions image_block_positions(int64_t start, int64_t grid_h, int64_t grid_w,
                                          const PreprocConfig &cfg, int64_t max_positions) {
    cfg.validate();
    STRIX_CHECK(start >= 0, "image_block_positions: block start is ", start, ", expected >= 0");
    const int64_t m = cfg.spatial_merge_size;
    STRIX_CHECK(grid_h >= m && grid_w >= m && grid_h % m == 0 && grid_w % m == 0,
                "image_block_positions: grid ", grid_h, "x", grid_w, ", expected both >= ", m,
                " and multiples of the merge ", m);
    const int64_t mh = grid_h / m, mw = grid_w / m;
    const int64_t block_end = start + std::max(grid_h, grid_w) / 2;  // one past the block's max position
    if (max_positions > 0)
        STRIX_CHECK(block_end - 1 < max_positions, "image_block_positions: the block at start ", start,
                    " with grid ", grid_h, "x", grid_w, " reaches position ", block_end - 1, ", expected "
                    "below the model's ", max_positions);
    ImageBlockPositions out;
    out.merged_tokens = mh * mw;
    out.next_start = block_end;
    out.pos3.reserve((size_t)(out.merged_tokens * 3));
    for (int64_t cell = 0; cell < out.merged_tokens; ++cell) {
        const int64_t cr = cell / mw, cc = cell % mw;
        out.pos3.push_back(start);
        out.pos3.push_back(start + cr);
        out.pos3.push_back(start + cc);
    }
    return out;
}

ImageBlockPositions image_block_positions(int64_t start, const Preprocessed &img, const PreprocConfig &cfg,
                                          int64_t max_positions) {
    STRIX_CHECK(img.grid_t == 1, "image_block_positions: grid_t is ", img.grid_t,
                ", expected 1 (images; video is not built)");
    return image_block_positions(start, img.grid_h, img.grid_w, cfg, max_positions);
}

}  // namespace strix::vision
