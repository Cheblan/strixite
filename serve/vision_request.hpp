#pragma once

// The serving-side image request path: everything between the OpenAI request's image parts and the
// Engine::GenerationRequest - data URL parse (base64), the image bytes through the preprocessor, the
// id-stream expansion of the single template pad per image, the (t, h, w) triples walk (the
// get_rope_index rule), and the cache-key ids image prompts are matched and stored under (image_pad
// runs replaced by content-derived pseudo ids - a prefix hit then means the same images, same places).
//
// It builds exactly the ImageInput / set_positions consumer contract, never edits it. Errors are
// strix::Error with expected vs found; the HTTP layer words them as 400s (it wraps every call below - a
// bad image is the client's error, never the engine's).

#include "serve/engine.hpp"      // VisionPrefill, GenerationRequest
#include "serve/vision_preprocess.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strix::vision {

// The served vision configuration: the preprocessor values (from the checkpoint's own files) + the tower file
// alongside the served weights. Built once at startup (strix_server); the HTTP layer only reads it.
struct ServedVision {
    PreprocConfig cfg;
    std::string vision_weights;  // vision.strixw ("" only before configure - a served engine has one)
    int32_t image_token_id = 0;  // config.json image_token_id (the template's image_pad row id)
};

// Read the preprocessor values from <checkpoint>/preprocessor_config.json and <checkpoint>/config.json's
// vision_config (every field the producer needs; disagreements between the two files refuse, naming
// both). max_patches is the engine's attention cap (kernels::kMaxVisionSeq) - the resize clamp.
ServedVision load_vision_config(const std::string &preprocessor_config_path, const std::string &config_json_path,
                                int64_t max_patches);

// One request image: the decoded PNG / JPEG bytes + the media type the data URL declared (kept for the log).
struct ParsedImage {
    std::vector<uint8_t> bytes;
    std::string mime;  // lower-case, e.g. "image/png"
};

// A "data:<mime>;base64,<payload>" URL (the only form accepted; remote URLs refuse, the server fetches nothing).
// Refusals (expected vs found, `where` names the request field): not a data URL (an http(s) URL says so
// explicitly), a media type other than the served ones (png / jpeg / gif / bmp / x-tga / webp), malformed
// base64 (alphabet, padding, trailing bits), an empty payload. The payload is decoded here; the image bytes
// themselves are validated by the preprocessor (magic, decode, channels) - decode_image refuses the rest
// with its own messages. The bytes' own magic, not this label, decides the decoder.
ParsedImage parse_data_url(const std::string &url, const std::string &where);

// Per image, in stream order: what the walk and the engine need.
struct ServedImage {
    VisionPrefill prefill;           // pixel_values + grid (the preprocessor's output)
    int64_t merged_tokens = 0;       // grid_h * grid_w / merge^2 - the pad expansion count
    int64_t merge = 0;               // the merge side (the triples' block grid divides by this)
    int64_t row_start = -1;          // set by plan_vision_prompt (walk)
    int64_t rows = 0;                // = merged_tokens once planned
    uint64_t digest = 0;             // over the bytes (the cache-key pseudo ids)
};

// An image prompt made ready for the engine. The walk (the get_rope_index rule, pinned by the vision_logit
// get_rope_index rule): text rows take (p, p, p) with p the running position; an image block at p takes per merged cell
// (p, p + cell_row, p + cell_col) and the walk continues at p + max(grid_h, grid_w) / 2.
struct VisionPromptPlan {
    std::vector<int64_t> pos3;           // 3 rows per prompt row (GenerationRequest::image_pos3)
    std::vector<int32_t> cache_prompt;   // prompt with each image's pad run replaced by its pseudo-id runs
    int64_t merged_rows_total = 0;
    int64_t rope_positions_reached = 0;  // the walk's end - checked against the model's max_positions
};

// The grids of the images this server has decoded, keyed by the digest of their bytes (ServedImage::digest - the
// same key the prompt cache matches image rows on). An agent loop sends its whole history every step, so every
// earlier screenshot arrives again with each request, and those blocks are almost always behind the resume point:
// prepare_images takes their grid from here and decodes nothing. The grid came from a real decode of the same bytes
// under the same PreprocConfig (fixed for the server's life), so it is exact - no header parse, no second rule.
// Thread-safe (HTTP threads share one). Bounded: past kMaxEntries the oldest entry goes (a forgotten image is
// decoded again, nothing worse).
class DecodeMemo {
public:
    struct Grid {
        int64_t grid_t = 0, grid_h = 0, grid_w = 0, merged_tokens = 0;
    };
    static constexpr size_t kMaxEntries = 65536;  // ~3 MB of map; far past any session's image count
    bool find(uint64_t digest, Grid &out) const;
    void remember(uint64_t digest, const Grid &grid);
    size_t size() const;

private:
    mutable std::mutex mu_;
    std::unordered_map<uint64_t, Grid> grids_;
    std::deque<uint64_t> order_;  // insertion order, for the bound
};

// Preprocess every image (decode -> smart-resize -> normalize -> pack) and plan the prompt. `prompt` is the
// tokenized template text with the placeholders ALREADY expanded (expand_image_pads); image_pad_id is the id of
// the vision triplet (vision_start / image_pad / vision_end; image_pad is config.json's image_token_id).
// Refusals: an image whose grid would exceed the attention cap
// (the producer's own checks - the resize clamp prevents it, this is the belt), a prompt whose image_pad count
// differs from images.size() (checked again here), a walk past max_positions.
// The ServedImage list comes back aligned with the pads' stream order (the walk's order - which the template
// and the parser both fix to message order; a mismatch is a walk refusal, never a silent reorder).
// prepare_images: with a memo, an image whose digest it holds is not decoded - its pixel_values stay empty and
// its VisionPrefill::decode keeps the bytes to build them if the prefill reaches the block after all (the engine
// calls it). Every other image is decoded here, so a bad image still refuses on this thread (a 400), and its grid
// is remembered once the decode succeeds. memo == nullptr: every image decoded (tools, tests).
std::vector<ServedImage> prepare_images(std::vector<ParsedImage> parsed, const ServedVision &vision,
                                        DecodeMemo *memo = nullptr);
VisionPromptPlan plan_vision_prompt(const std::vector<int32_t> &prompt, std::vector<ServedImage> &images,
                                    const PreprocConfig &cfg, int32_t image_pad_id, int64_t max_positions);

// Token-level expansion: every image_pad occurrence (stream order) replaced by that image's merged_tokens
// copies - the processor's contract. `merged_per_image` must
// have exactly one entry per pad occurrence (expected vs found names both counts).
std::vector<int32_t> expand_image_pads(const std::vector<int32_t> &prompt, int32_t image_pad_id,
                                       const std::vector<int64_t> &merged_per_image);

// How many image preprocesses (decode -> resize -> pack, all on the requesting HTTP thread) may run at once.
// Each one's worst case is a few hundred MB of transient host RAM - the per-image decode cap
// (PreprocConfig::max_decode_bytes) plus the resized planar / packed copies, which the patch cap keeps small
// next to the decode - and they add up across concurrent requests while the served model holds most of the
// machine's RAM, where low free memory hard-hangs the box rather than dying cleanly to the OOM killer. So a
// new image request whose preprocess would exceed the limit is REFUSED, named and retryable (config
// vision-preprocess-limit) - a loud 503, never a queue that succeeds into a hang. The refusal is the feature.
class PreprocessGate {
public:
    // One reserved slot: releases itself back to the gate when it dies (move-only, so the holder's every
    // exit path - success, refusal, exception - frees the memory it reserved).
    class Slot {
    public:
        Slot() = default;
        ~Slot() { release(); }
        Slot(Slot && other) noexcept : gate_(other.gate_) { other.gate_ = nullptr; }
        Slot & operator=(Slot && other) noexcept { if (this != &other) { release(); gate_ = other.gate_; other.gate_ = nullptr; } return *this; }
        Slot(const Slot &) = delete;
        Slot & operator=(const Slot &) = delete;

    private:
        friend class PreprocessGate;
        explicit Slot(PreprocessGate *gate) : gate_(gate) {}
        void release() { if (gate_ != nullptr) { gate_->in_flight_.fetch_sub(1); gate_ = nullptr; } }
        PreprocessGate *gate_ = nullptr;
    };

    // Take one of the `limit` slots; throws (expected vs found: in flight vs the limit) when all are held.
    Slot acquire(int64_t limit);
    int64_t in_flight() const { return in_flight_.load(); }

private:
    std::atomic<int64_t> in_flight_{0};
};

}  // namespace strix::vision
