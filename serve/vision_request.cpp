// The serving-side image request path - see serve/vision_request.hpp for the contract. Runs on the
// HTTP thread (the engine worker never decodes or packs).

#include "serve/vision_request.hpp"

#include "common/check.hpp"
#include "formats/strixw.hpp"  // strix_hash64 (the same 64-bit hash the prompt cache's keys use)
#include "serve/json.hpp"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace strix::vision {

namespace {

std::string to_lower(std::string s) {
    for (char &c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::string read_file(const std::string &path, const char *fn) {
    std::ifstream f(path, std::ios::binary);
    STRIX_CHECK(f.good(), fn, ": can't open '", path, "'");
    std::ostringstream ss;
    ss << f.rdbuf();
    STRIX_CHECK(!ss.str().empty(), fn, ": '", path, "' is empty");
    return ss.str();
}

double json_number(const json::Value &v, const std::string &path) {
    STRIX_CHECK(v.is_number(), "'", path, "' must be a number, got ", json::type_name(v.type()));
    return v.as_double(path);
}

int64_t json_int(const json::Value &v, const std::string &path) {
    STRIX_CHECK(v.is_int(), "'", path, "' must be an integer, got ", json::type_name(v.type()));
    return v.as_int(path);
}

// The 3-channel mean / std arrays (image_mean / image_std).
void json_vec3(const json::Value &v, const std::string &path, float out[3]) {
    STRIX_CHECK(v.is_array() && v.as_array(path).size() == 3, "'", path, "' must be a 3-element array, got ",
                json::type_name(v.type()));
    for (int c = 0; c < 3; ++c)
        out[c] = (float)json_number(v.as_array(path)[(size_t)c], path + "[" + std::to_string(c) + "]");
}

// base64, strict RFC 4648: standard alphabet, length a multiple of 4, '=' only as final padding, no
// whitespace, no non-zero dropped bits. A mangled payload is a named 400 - never half-decoded bytes
// handed to the image decoder (the no-fallback rule).
std::vector<uint8_t> base64_decode(const std::string &in, const std::string &where) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    STRIX_CHECK(in.size() >= 4 && in.size() % 4 == 0, where, ": base64 payload of ", in.size(),
                " characters, expected a positive multiple of 4 (strict RFC 4648, padding included)");
    size_t pad = 0;
    if (in.back() == '=') pad = in.size() >= 2 && in[in.size() - 2] == '=' ? 2 : 1;
    std::vector<uint8_t> out(in.size() / 4 * 3 - pad);
    size_t o = 0;
    for (size_t i = 0; i < in.size(); i += 4) {
        const bool last = i + 4 == in.size();
        const size_t drop = last ? pad : 0;  // octets dropped from this quartet's tail
        uint32_t bits = 0;
        for (size_t k = 0; k < 4; ++k) {
            const char c = in[i + k];
            uint32_t v = 0;
            if (c == '=')
                STRIX_CHECK(drop > 0 && k >= 4 - drop, where, ": base64 padding '=' at character ", i + k,
                            ", expected it only as the final ", drop == 2 ? "two" : "one", " character(s)");
            else {
                const int x = val(c);
                STRIX_CHECK(x >= 0, where, ": base64 payload holds '", std::string(1, c), "' at character ",
                            i + k, ", expected the RFC 4648 alphabet (A-Z a-z 0-9 + /)");
                v = (uint32_t)x;
            }
            bits = (bits << 6) | v;
        }
        const size_t bytes_here = 3 - drop;
        if (drop) {  // the unused low bits of the last value sextet(s) must be zero: pad 1 -> the last
                     // sextet keeps 2 bits (4 unused at bit 2..5), pad 2 -> the second-to-last keeps 4
                     // (2 unused at bits 12..13; the '=' sextet is decoded as 0)
            const uint32_t mask = drop == 1 ? 0xc0u : 0xf000u;
            STRIX_CHECK((bits & mask) == 0, where, ": base64 payload's last quartet has non-zero unused bits "
                        "(mangled tail)");
        }
        for (size_t b = 0; b < bytes_here; ++b) out[o++] = (uint8_t)(bits >> (8 * (2 - b)));
    }
    STRIX_CHECK(o == out.size(), where, ": base64 decode produced ", o, " bytes, expected ", out.size(),
                " (internal)");
    return out;
}

// One int32 pseudo id from a 64-bit hash slice, strictly negative (real token ids are >= 0).
// -1 - x rather than -(x + 1): the latter negates INT_MIN when the low 31 bits are all ones (UB).
int32_t pseudo_id(uint64_t h) {
    return -1 - (int32_t)(h & 0x7fffffffu);  // -1 .. INT_MIN
}

}  // namespace

ServedVision load_vision_config(const std::string &preprocessor_config_path, const std::string &config_json_path,
                                int64_t max_patches) {
    const char *fn = "load_vision_config";
    const json::Value pp = json::Value::parse(read_file(preprocessor_config_path, fn));
    const json::Value cfg = json::Value::parse(read_file(config_json_path, fn));
    const json::Value *vcfg = cfg.find("vision_config");
    STRIX_CHECK(vcfg != nullptr && vcfg->is_object(), fn, ": '", config_json_path, "' has no vision_config "
                "object (the checkpoint's config.json - a towerless conversion has no image path)");
    const auto need_pp = [&](const char *k) -> const json::Value & {
        const json::Value *v = pp.find(k);
        STRIX_CHECK(v != nullptr, fn, ": '", preprocessor_config_path, "' has no '", k, "'");
        return *v;
    };
    ServedVision out;
    PreprocConfig &c = out.cfg;
    json_vec3(need_pp("image_mean"), "preprocessor.image_mean", c.mean);
    json_vec3(need_pp("image_std"), "preprocessor.image_std", c.std);
    const json::Value &size = need_pp("size");
    STRIX_CHECK(size.is_object(), fn, ": preprocessor 'size' must be an object, got ",
                json::type_name(size.type()));
    const json::Value *shortest = size.find("shortest_edge"), *longest = size.find("longest_edge");
    STRIX_CHECK(shortest && longest, fn, ": preprocessor 'size' lacks shortest_edge / longest_edge");
    c.min_pixels = json_int(*shortest, "preprocessor.size.shortest_edge");
    c.max_pixels = json_int(*longest, "preprocessor.size.longest_edge");
    const struct { const char *pp_key, *vc_key; int64_t PreprocConfig::*field; } shared[] = {
        {"patch_size", "patch_size", &PreprocConfig::patch_size},
        {"temporal_patch_size", "temporal_patch_size", &PreprocConfig::temporal_patch_size},
        {"merge_size", "spatial_merge_size", &PreprocConfig::spatial_merge_size},
    };
    for (const auto &f : shared) {
        const int64_t a = json_int(need_pp(f.pp_key), std::string("preprocessor.") + f.pp_key);
        const json::Value *b = vcfg->find(f.vc_key);
        STRIX_CHECK(b != nullptr, fn, ": vision_config has no '", f.vc_key, "'");
        const int64_t bv = json_int(*b, std::string("vision_config.") + f.vc_key);
        STRIX_CHECK(a == bv, fn, ": preprocessor_config.", f.pp_key, " = ", a, ", vision_config.", f.vc_key,
                    " = ", bv, " - the checkpoint's files disagree");
        c.*(f.field) = a;
    }
    const json::Value *ich = vcfg->find("in_channels");
    c.in_channels = ich ? json_int(*ich, "vision_config.in_channels") : 3;
    c.max_patches = max_patches;
    c.validate();
    const json::Value *pad = cfg.find("image_token_id");
    STRIX_CHECK(pad != nullptr && pad->is_int(), fn, ": config.json has no integer image_token_id (the "
                "template's image_pad row id)");
    const int64_t pad_id = json_int(*pad, "image_token_id");
    STRIX_CHECK(pad_id >= 0 && pad_id <= std::numeric_limits<int32_t>::max(), fn, ": config.json image_token_id = ",
                pad_id, ", expected a token id in 0..", std::numeric_limits<int32_t>::max());
    out.image_token_id = (int32_t)pad_id;
    return out;
}

ParsedImage parse_data_url(const std::string &url, const std::string &where) {
    ParsedImage out;
    if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0)
        STRIX_FAIL(where, ": a remote image URL '", url.substr(0, 48),
                   "...' - only data: URLs are served (the server fetches nothing)");
    if (url.rfind("data:", 0) != 0)
        STRIX_FAIL(where, "'", url.substr(0, 48), "...' is not a data: URL (expected "
                   "data:image/<png|jpeg|gif|bmp|x-tga|webp>;base64,<payload>)");
    const size_t comma = url.find(',');
    STRIX_CHECK(comma != std::string::npos, where, ": data URL without the ',' that ends its header");
    std::string header = to_lower(url.substr(5, comma - 5));
    STRIX_CHECK(header.size() > 7 && header.compare(header.size() - 7, 7, ";base64") == 0, where,
                ": data URL header '", header.substr(0, 64), "' does not end in ';base64' (only base64 "
                "payloads are accepted)");
    header.erase(header.size() - 7);  // drop ";base64"; any further parameters (charset, ...) are dropped too
    const size_t semi = header.find(';');
    out.mime = semi == std::string::npos ? header : header.substr(0, semi);
    // Common aliases, normalized. The bytes' own magic decides the decoder (decode_image), not this label.
    if (out.mime == "image/jpg") out.mime = "image/jpeg";
    if (out.mime == "image/x-ms-bmp" || out.mime == "image/x-bmp") out.mime = "image/bmp";
    if (out.mime == "image/tga" || out.mime == "image/x-targa" || out.mime == "image/targa") out.mime = "image/x-tga";
    STRIX_CHECK(out.mime == "image/png" || out.mime == "image/jpeg" || out.mime == "image/gif" ||
                    out.mime == "image/bmp" || out.mime == "image/x-tga" || out.mime == "image/webp",
                where, ": media type '", out.mime,
                "' is not served (image/png, image/jpeg, image/gif, image/bmp, image/x-tga or image/webp)");
    const std::string payload = url.substr(comma + 1);
    STRIX_CHECK(!payload.empty(), where, ": data URL with an empty base64 payload");
    out.bytes = base64_decode(payload, where);
    STRIX_CHECK(!out.bytes.empty(), where, ": data URL decoded to 0 bytes");
    return out;
}

bool DecodeMemo::find(uint64_t digest, Grid &out) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = grids_.find(digest);
    if (it == grids_.end()) return false;
    out = it->second;
    return true;
}

void DecodeMemo::remember(uint64_t digest, const Grid &grid) {
    STRIX_CHECK(grid.grid_t >= 1 && grid.grid_h >= 1 && grid.grid_w >= 1 && grid.merged_tokens >= 1,
                "DecodeMemo::remember: grid ", grid.grid_t, "x", grid.grid_h, "x", grid.grid_w, " with ",
                grid.merged_tokens, " merged tokens, expected every field >= 1 (a decoded image's grid)");
    std::lock_guard<std::mutex> lock(mu_);
    if (!grids_.emplace(digest, grid).second) return;
    order_.push_back(digest);
    while (order_.size() > kMaxEntries) {
        grids_.erase(order_.front());
        order_.pop_front();
    }
}

size_t DecodeMemo::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return grids_.size();
}

std::vector<ServedImage> prepare_images(std::vector<ParsedImage> parsed, const ServedVision &vision,
                                        DecodeMemo *memo) {
    std::vector<ServedImage> out;
    out.reserve(parsed.size());
    for (size_t i = 0; i < parsed.size(); ++i) {
        STRIX_CHECK(!parsed[i].bytes.empty(), "prepare_images: image ", i + 1, " of ", parsed.size(),
                    " has 0 bytes (parse_data_url refuses those - internal)");
        ServedImage img;
        img.merge = vision.cfg.spatial_merge_size;
        img.digest = strix_hash64(parsed[i].bytes.data(), parsed[i].bytes.size());
        DecodeMemo::Grid known;
        if (memo != nullptr && memo->find(img.digest, known)) {
            img.prefill.grid_t = known.grid_t, img.prefill.grid_h = known.grid_h, img.prefill.grid_w = known.grid_w;
            img.merged_tokens = known.merged_tokens;
            auto bytes = std::make_shared<const std::vector<uint8_t>>(std::move(parsed[i].bytes));
            img.prefill.decode = [bytes, cfg = vision.cfg, known, digest = img.digest]() {
                Preprocessed pre = preprocess_image(*bytes, cfg);
                STRIX_CHECK(pre.grid_t == known.grid_t && pre.grid_h == known.grid_h && pre.grid_w == known.grid_w &&
                                pre.merged_tokens == known.merged_tokens,
                            "prepare_images: image ", digest, " decoded at prefill to grid ", pre.grid_t, "x",
                            pre.grid_h, "x", pre.grid_w, " (", pre.merged_tokens, " merged), its first decode gave ",
                            known.grid_t, "x", known.grid_h, "x", known.grid_w, " (", known.merged_tokens,
                            ") - the same bytes must give the same grid");
                return std::move(pre.pixel_values);
            };
            out.push_back(std::move(img));
            continue;
        }
        Preprocessed pre = preprocess_image(parsed[i].bytes, vision.cfg);  // refuses by name (stb's
        // failure reason, the caps, the grid rule - all expected-vs-found, with the request layer naming
        // which image part failed)
        img.prefill.pixel_values = std::move(pre.pixel_values);
        img.prefill.grid_t = pre.grid_t, img.prefill.grid_h = pre.grid_h, img.prefill.grid_w = pre.grid_w;
        img.merged_tokens = pre.merged_tokens;
        if (memo != nullptr) memo->remember(img.digest, {pre.grid_t, pre.grid_h, pre.grid_w, pre.merged_tokens});
        out.push_back(std::move(img));
    }
    return out;
}

std::vector<int32_t> expand_image_pads(const std::vector<int32_t> &prompt, int32_t image_pad_id,
                                        const std::vector<int64_t> &merged_per_image) {
    int64_t pads = 0;
    for (const int32_t id : prompt) pads += id == image_pad_id;
    STRIX_CHECK(pads == (int64_t)merged_per_image.size(), "expand_image_pads: the tokenized prompt carries ",
                pads, " image_token_id (", image_pad_id, ") rows, the request sent ", merged_per_image.size(),
                " image(s) - the template's <|image_pad|> count must equal the image parts'");
    std::vector<int32_t> out;
    out.reserve(prompt.size() + 1024);
    size_t next = 0;
    for (const int32_t id : prompt) {
        if (id != image_pad_id) {
            out.push_back(id);
            continue;
        }
        const int64_t n = merged_per_image[next++];
        STRIX_CHECK(n >= 1, "expand_image_pads: image ", next, " expands to ", n,
                    " tokens, expected >= 1 (a grid of at least merge^2 patches)");
        out.insert(out.end(), (size_t)n, image_pad_id);
    }
    return out;
}

VisionPromptPlan plan_vision_prompt(const std::vector<int32_t> &prompt, std::vector<ServedImage> &images,
                                    const PreprocConfig &cfg, int32_t image_pad_id, int64_t max_positions) {
    VisionPromptPlan plan;
    int64_t cur = 0;  // the walk's position (the design note's current_pos)
    size_t next_img = 0;
    for (int64_t r = 0; r < (int64_t)prompt.size();) {
        if (prompt[(size_t)r] != image_pad_id) {  // a text row: all three sections at the running position
            plan.pos3.insert(plan.pos3.end(), {cur, cur, cur});
            plan.cache_prompt.push_back(prompt[(size_t)r]);  // real id (a pad can only come from a run below)
            ++cur, ++r;
            continue;
        }
        int64_t run_end = r;
        while (run_end < (int64_t)prompt.size() && prompt[(size_t)run_end] == image_pad_id) ++run_end;
        const int64_t run = run_end - r;
        STRIX_CHECK(next_img < images.size(), "plan_vision_prompt: the prompt has image_token_id rows beyond "
                    "its ", images.size(), " image(s) - expand_image_pads and the walk must share one list");
        ServedImage &img = images[next_img++];
        STRIX_CHECK(run == img.merged_tokens, "plan_vision_prompt: image ", next_img, " has a run of ", run,
                    " image_token_id rows, its grid (", img.prefill.grid_h, "x", img.prefill.grid_w,
                    ") gives ", img.merged_tokens, " merged tokens");
        const auto blk = image_block_positions(cur, img.prefill.grid_h, img.prefill.grid_w, cfg, max_positions);
        img.row_start = r, img.rows = run;
        plan.pos3.insert(plan.pos3.end(), blk.pos3.begin(), blk.pos3.end());
        cur = blk.next_start;
        // Cache-key ids for the block: two pseudo-id runs derived from the image bytes (real ids are never
        // negative, so a key stream can only ever match a prompt with the same images in the same places).
        const int32_t a = pseudo_id(img.digest), b = pseudo_id(img.digest ^ 0x9e3779b97f4a7c15ull);
        const int64_t half = (run + 1) / 2;
        plan.cache_prompt.insert(plan.cache_prompt.end(), (size_t)half, a);
        plan.cache_prompt.insert(plan.cache_prompt.end(), (size_t)(run - half), b);
        plan.merged_rows_total += run;
        r = run_end;
    }
    STRIX_CHECK(next_img == images.size(), "plan_vision_prompt: ", images.size(), " image(s) sent, ", next_img,
                " image_token_id runs found in the prompt - a pad was lost between render and tokenize");
    plan.rope_positions_reached = cur;
    STRIX_CHECK((int64_t)plan.pos3.size() == 3 * (int64_t)prompt.size(),
                "plan_vision_prompt: the walk built ", plan.pos3.size() / 3, " rows for a prompt of ",
                prompt.size(), " (internal)");
    return plan;
}

PreprocessGate::Slot PreprocessGate::acquire(int64_t limit) {
    STRIX_CHECK(limit >= 1, "PreprocessGate::acquire: limit ", limit, ", expected >= 1 (config vision-preprocess-limit)");
    // Claim first, then check: two callers can both see room, but the counter they incremented is the truth -
    // the loser of the race is over the limit and un-claims. (A CAS loop would work too; the fetch_add and
    // the rare fetch_sub are simpler to read than the loop and this is not a hot path.)
    const int64_t in_flight_before = in_flight_.fetch_add(1);
    if (in_flight_before + 1 > limit) {
        in_flight_.fetch_sub(1);
        STRIX_FAIL("PreprocessGate: all ", limit, " image preprocess slots are held (config vision-preprocess-limit = ",
                   limit, ") - the decode / resize runs on the request thread and each one costs a few hundred MB of "
                   "host RAM; retry the image request shortly");
    }
    return Slot(this);
}

}  // namespace strix::vision
