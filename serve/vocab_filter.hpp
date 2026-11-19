#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strix {

class Tokenizer;

namespace vocab_filter {

enum Section : uint32_t {
    Core     = 1 << 0,
    Latin    = 1 << 1,
    Cyrillic = 1 << 2,
    Greek    = 1 << 3,
    Digits   = 1 << 4,
    Punct    = 1 << 5,
    Code     = 1 << 6,
    Cjk      = 1 << 7,
    Arabic   = 1 << 8,
    Asian    = 1 << 9,
    Other    = 1 << 10,
    All      = 0x7FF
};

// Presets
constexpr uint32_t kPresetAll = Section::All;
constexpr uint32_t kPresetEnRu = Section::Core | Section::Latin | Section::Cyrillic | Section::Greek |
                                 Section::Digits | Section::Punct | Section::Code;
constexpr uint32_t kPresetEn = Section::Core | Section::Latin | Section::Greek | Section::Digits |
                               Section::Punct | Section::Code;
constexpr uint32_t kPresetNoCjk = Section::All & ~Section::Cjk;

// Parse filter specification string ("all", "en+ru", "en", "no-cjk", or comma-separated list of section names).
// Crashes immediately with fatal error if 'core' is missing when not "all".
uint32_t parse_filter(const std::string &spec);

// Returns true if filter covers all tokens (no filtering needed).
inline bool is_all(uint32_t mask) {
    return mask == Section::All;
}

// Filter tokenizer tokens up to max_vocab (e.g. min(tok.size(), lm_head.N())).
// Returns compact_to_vocab: list of original token IDs retained in the compact vocabulary.
std::vector<int32_t> filter_vocabulary(const Tokenizer &tok, uint32_t mask, int64_t max_vocab);

}  // namespace vocab_filter
}  // namespace strix
