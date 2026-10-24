#pragma once

// Offline converter: Qwen3.8-Flash-Next HF safetensors -> weights.strixw (formats/strixw.hpp; layout A by default).
// Plans every output tensor from the checkpoint headers alone (merge groups from formats/merge_plan, the HC
// fold, chunk-major W_up, the router in BF16, small tensors widened to F32), then streams the sources
// (pread in ~256 MB chunks, the next chunk read while this one is quantized on all cores, page cache
// dropped behind) and writes each tensor as soon as it's done. Verbose by design: the log explains each
// transform the first time it appears and reports per-tensor size, bits per weight, sampled quantization
// error and time, and per-layer progress with throughput and an ETA.

#include "formats/strixw.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace strix {

// The vision tower's config.json "vision_config" fields the converter validates the checkpoint's
// model.visual.* tensors against. Filled from config.json by the caller (the tool, the test) - this
// library stays JSON-free; the raw JSON goes into the output's index as-is. The only precision choice
// here is that everything is copied as exact BF16 (the tower runs
// once per image; quantizing it saves memory nobody asked about and buys no time).
struct VisionDims {
    int64_t depth = 0, hidden_size = 0, num_heads = 0, intermediate_size = 0, in_channels = 0, patch_size = 0,
            temporal_patch_size = 0, spatial_merge_size = 0, out_hidden_size = 0, num_position_embeddings = 0;
    std::string hidden_act;  // must be "gelu_pytorch_tanh" (what this checkpoint has; anything else is refused)
};

struct ConvertOptions {
    std::string src_dir;      // HF checkpoint directory (*.safetensors + model.safetensors.index.json)
    std::string out_dir;      // created if missing; must not already hold weights.strixw
    std::string git;          // converter git hash, recorded in the metadata
    // Per-class formats: "default=q4g64" is layout A; e.g. ",gdn_in=q8g64" puts a class
    // at Q8 (only where a kernel exists).
    std::string layout = "default=q4g64";
    int threads = 0;          // 0: all hardware threads
    std::vector<int64_t> layers;  // decoder layers to convert; empty = all
    bool globals = true;          // embedding, LM head, final HC mixer
    bool mtp = true;              // the MTP head
    // Separate MTP: write out_dir/mtp.strixw as its own strixw file and omit mtp.* tensors from weights.strixw.
    bool mtp_separate = false;
    // MTP only: write exactly out_dir/mtp.strixw and nothing else (never touches existing weights.strixw).
    bool mtp_only = false;
    // A safetensors file whose tensors replace the checkpoint's same-named mtp.* tensors (a fine-tuned draft head):
    // every tensor in it must be an mtp.* tensor the checkpoint has, same dtype and shape. The trunk still comes from
    // src_dir, so source_fingerprint stays the checkpoint's (it pairs with the n-gram table); the metadata records
    // the file and its fingerprint as mtp_source. Empty: none.
    std::string mtp_from;
    bool plan_only = false;       // log the plan and stop (reads headers only)
    FILE *console = stderr;       // log sink besides out_dir/convert.log (nullptr: file only)
    // Vision tower: additionally write out_dir/vision.strixw,
    // every model.visual.* tensor as an exact BF16 copy, validated against vision_dims. weights.strixw is
    // unchanged either way (the text file never carries the tower, so existing downloads stay valid).
    bool vision = false;
    // Tower only: write exactly that file and nothing else - the path to add a tower to an output
    // directory converted before --vision existed (the text plan is not read, its refusal of an existing
    // weights.strixw does not apply; the tower file's own existence still stops it).
    bool tower_only = false;
    VisionDims vision_dims;
    std::string vision_config_json;  // the raw "vision_config" object text, stored in the new file's index
};

enum class ConvertOp { Q4Merge, Q4FoldHC, Q4ChunkMajor, BF16Rows, F32Widen, I64Copy };
const char *convert_op_name(ConvertOp op);

struct ConvertStep {
    StrixwTensor out;                  // name, encoding, shape, group size, experts, parts
    ConvertOp op = ConvertOp::F32Widen;
    std::vector<std::string> sources;  // Q4Merge / BF16Rows: the parts; Q4FoldHC: down, inject (or ""), norm
    int64_t unit = -1;                 // decoder layer index, -1 globals, -2 MTP (for progress grouping)
};

struct ConvertPlan {
    std::vector<ConvertStep> steps;
    std::vector<std::pair<std::string, std::string>> skipped;  // source tensor, why
    int64_t layers_total = 0;
    bool shared_split = false;  // shared expert written separately (format differs from the routed experts')
};

// Summary of a finished (or planned) run.
struct ConvertResult {
    std::string weights_path;
    std::string vision_path;  // "" unless ConvertOptions::vision
    std::string mtp_path;     // "" unless ConvertOptions::mtp_separate or mtp_only
    uint64_t source_bytes = 0, output_bytes = 0;
    size_t tensors = 0;
    double seconds = 0;
};

ConvertResult convert_qwen4exp(const ConvertOptions &opt);

// The source checkpoint's fingerprint as the converter records it (strixw metadata source_fingerprint): a hash of
// the index JSON and every shard's name, size and header. Files converted from the same checkpoint carry the same.
std::string checkpoint_fingerprint(const std::string &dir);

}  // namespace strix
