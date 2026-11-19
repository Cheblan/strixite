#include "serve/vocab_filter.hpp"
#include "serve/tokenizer.hpp"
#include "common/check.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace strix::vocab_filter {

namespace {

std::string trim(const std::string &s) {
    const char *ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

std::string to_lower(std::string s) {
    for (char &c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

bool decode_utf8_safe(std::string_view s, std::vector<uint32_t> &out) {
    out.clear();
    const uint8_t *p = reinterpret_cast<const uint8_t *>(s.data());
    const size_t len = s.size();
    size_t i = 0;
    while (i < len) {
        uint8_t c = p[i];
        if (c < 0x80) {
            out.push_back(c);
            i++;
        } else if ((c >> 5) == 0x06) {
            if (i + 1 >= len || (p[i + 1] >> 6) != 0x02) return false;
            uint32_t cp = ((c & 0x1F) << 6) | (p[i + 1] & 0x3F);
            out.push_back(cp);
            i += 2;
        } else if ((c >> 4) == 0x0E) {
            if (i + 2 >= len || (p[i + 1] >> 6) != 0x02 || (p[i + 2] >> 6) != 0x02) return false;
            uint32_t cp = ((c & 0x0F) << 12) | ((p[i + 1] & 0x3F) << 6) | (p[i + 2] & 0x3F);
            out.push_back(cp);
            i += 3;
        } else if ((c >> 3) == 0x1E) {
            if (i + 3 >= len || (p[i + 1] >> 6) != 0x02 || (p[i + 2] >> 6) != 0x02 || (p[i + 3] >> 6) != 0x02) return false;
            uint32_t cp = ((c & 0x07) << 18) | ((p[i + 1] & 0x3F) << 12) | ((p[i + 2] & 0x3F) << 6) | (p[i + 3] & 0x3F);
            out.push_back(cp);
            i += 4;
        } else {
            return false;
        }
    }
    return true;
}

uint32_t classify_codepoint(uint32_t cp) {
    // 1. Core whitespace
    if (cp == 9 || cp == 10 || cp == 13 || cp == 32) return Section::Core;

    // 2. Digits
    if ((cp >= 0x30 && cp <= 0x39) || (cp >= 0xFF10 && cp <= 0xFF19)) return Section::Digits;

    // 3. Punctuation, Math, Typography, Currency
    if ((cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) ||
        (cp >= 0x5B && cp <= 0x60) || (cp >= 0x7B && cp <= 0x7E)) {
        return Section::Punct;
    }
    // Latin-1 Supplement punctuation and symbols U+00A0..U+00BF («»°±·§©® etc.)
    if (cp >= 0x00A0 && cp <= 0x00BF) return Section::Punct;
    // General Punctuation U+2000..U+206F (em-dash, curly quotes, ellipsis, etc.)
    if (cp >= 0x2000 && cp <= 0x206F) return Section::Punct;
    // Currency Symbols U+20A0..U+20CF (₽, €, ₹, ₿, etc.)
    if (cp >= 0x20A0 && cp <= 0x20CF) return Section::Punct;
    // Letterlike Symbols, Arrows, Math Operators, Technical, Dingbats U+2100..U+27BF
    if (cp >= 0x2100 && cp <= 0x27BF) return Section::Punct;

    // 4. Latin
    if ((cp >= 0x41 && cp <= 0x5A) || (cp >= 0x61 && cp <= 0x7A)) return Section::Latin;
    // Latin-1 Supplement accented letters U+00C0..U+00FF
    if (cp >= 0x00C0 && cp <= 0x00FF) return Section::Latin;
    // Latin Extended-A, B, IPA, Modifiers U+0100..U+02FF
    if (cp >= 0x0100 && cp <= 0x02FF) return Section::Latin;

    // 5. Cyrillic
    if ((cp >= 0x0400 && cp <= 0x052F) || (cp >= 0x2DE0 && cp <= 0x2DFF) || (cp >= 0xA640 && cp <= 0xA69F)) {
        return Section::Cyrillic;
    }

    // 6. Greek & Coptic
    if ((cp >= 0x0370 && cp <= 0x03FF) || (cp >= 0x1F00 && cp <= 0x1FFF)) return Section::Greek;

    // 7. CJK (Hanzi, Kana, Hangul, Bopomofo)
    if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
        (cp >= 0x3000 && cp <= 0x30FF) || (cp >= 0x3100 && cp <= 0x312F) ||
        (cp >= 0xAC00 && cp <= 0xD7AF) || (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x3130 && cp <= 0x318F) ||
        (cp >= 0xFF00 && cp <= 0xFFEF)) {
        return Section::Cjk;
    }

    // 8. Arabic & Hebrew
    if ((cp >= 0x0600 && cp <= 0x06FF) || (cp >= 0x0750 && cp <= 0x077F) || (cp >= 0x08A0 && cp <= 0x08FF) ||
        (cp >= 0xFB50 && cp <= 0xFDFF) || (cp >= 0xFE70 && cp <= 0xFEFF) || (cp >= 0x0590 && cp <= 0x05FF)) {
        return Section::Arabic;
    }

    // 9. Asian (Indic, Thai, Lao, Tibetan, Myanmar, Khmer, Georgian, Ethiopic)
    if ((cp >= 0x0900 && cp <= 0x0EFF) || (cp >= 0x0F00 && cp <= 0x10FF) ||
        (cp >= 0x1200 && cp <= 0x137F) || (cp >= 0x1780 && cp <= 0x17FF)) {
        return Section::Asian;
    }

    // 10. Other
    return Section::Other;
}

}  // namespace

uint32_t parse_filter(const std::string &spec_raw) {
    const std::string spec = to_lower(trim(spec_raw));
    if (spec.empty() || spec == "all" || spec == "full") {
        return Section::All;
    }
    if (spec == "en+ru" || spec == "ru+en" || spec == "coding+ru") {
        return kPresetEnRu;
    }
    if (spec == "en" || spec == "coding" || spec == "english") {
        return kPresetEn;
    }
    if (spec == "no-cjk" || spec == "western") {
        return kPresetNoCjk;
    }

    // Parse comma-separated sections
    uint32_t mask = 0;
    std::stringstream ss(spec);
    std::string token;
    while (std::getline(ss, token, ',')) {
        const std::string s = to_lower(trim(token));
        if (s.empty()) continue;
        if (s == "core") mask |= Section::Core;
        else if (s == "latin") mask |= Section::Latin;
        else if (s == "cyrillic") mask |= Section::Cyrillic;
        else if (s == "greek") mask |= Section::Greek;
        else if (s == "digits") mask |= Section::Digits;
        else if (s == "punct") mask |= Section::Punct;
        else if (s == "code") mask |= Section::Code;
        else if (s == "cjk") mask |= Section::Cjk;
        else if (s == "arabic") mask |= Section::Arabic;
        else if (s == "asian") mask |= Section::Asian;
        else if (s == "other") mask |= Section::Other;
        else {
            STRIX_FAIL("vocab_filter::parse_filter: unknown section '", s,
                       "'; expected core,latin,cyrillic,greek,digits,punct,code,cjk,arabic,asian,other or preset (all, en+ru, en, no-cjk)");
        }
    }

    // Unconditional fatal check: core is mandatory
    if (!(mask & Section::Core)) {
        std::fprintf(stderr, "FATAL: --mtp-vocab-filter must include 'core' (special tokens and raw byte fallback are required for MTP stability)\n");
        std::abort();
    }

    return mask;
}

std::vector<int32_t> filter_vocabulary(const Tokenizer &tok, uint32_t mask, int64_t max_vocab) {
    if (is_all(mask)) {
        return {};  // Fast path: empty signifies full unfiltered vocabulary
    }

    if (!(mask & Section::Core)) {
        std::fprintf(stderr, "FATAL: --mtp-vocab-filter must include 'core' (special tokens and raw byte fallback are required for MTP stability)\n");
        std::abort();
    }

    std::vector<int32_t> compact_to_vocab;
    compact_to_vocab.reserve((size_t)max_vocab);

    std::vector<uint32_t> cps;
    for (int32_t id = 0; id < (int32_t)max_vocab; ++id) {
        const std::string &raw = tok.token_bytes(id);

        // 1. Raw byte singletons (256 bytes) and empty tokens belong to core
        if (raw.size() <= 1) {
            if (mask & Section::Core) compact_to_vocab.push_back(id);
            continue;
        }

        // 2. Special added tokens (<|...|>) belong to core
        if (raw.size() >= 3 && raw[0] == '<' && raw[1] == '|' && raw.back() == '>') {
            if (mask & Section::Core) compact_to_vocab.push_back(id);
            continue;
        }

        // 3. UTF-8 decode
        if (decode_utf8_safe(raw, cps)) {
            // Check for code indents (pure whitespace >= 2 chars)
            bool is_indent = true;
            for (uint32_t cp : cps) {
                if (cp != ' ' && cp != '\t') { is_indent = false; break; }
            }
            if (is_indent && (mask & (Section::Code | Section::Core))) {
                compact_to_vocab.push_back(id);
                continue;
            }

            // All-allowed rule: every character must belong to allowed categories
            bool allowed = true;
            for (uint32_t cp : cps) {
                const uint32_t cat = classify_codepoint(cp);
                if (!(cat & mask)) {
                    // Check if punct/code overlaps for common syntax tokens
                    if ((cat == Section::Punct) && (mask & Section::Code)) {
                        // Allow punctuation under code
                        continue;
                    }
                    allowed = false;
                    break;
                }
            }
            if (allowed) {
                compact_to_vocab.push_back(id);
            }
        } else {
            // Partial / incomplete UTF-8 chunk from BPE: classify by leading byte
            const uint8_t lead = (uint8_t)raw[0];
            if (0xD0 <= lead && lead <= 0xD1) {
                if (mask & Section::Cyrillic) compact_to_vocab.push_back(id);
            } else if (0xD8 <= lead && lead <= 0xDB) {
                if (mask & Section::Arabic) compact_to_vocab.push_back(id);
            } else if ((0xE4 <= lead && lead <= 0xE9) || (0xEA <= lead && lead <= 0xED)) {
                if (mask & Section::Cjk) compact_to_vocab.push_back(id);
            } else if (0x00 <= lead && lead <= 0x7F) {
                if (mask & Section::Core) compact_to_vocab.push_back(id);
            }
        }
    }

    return compact_to_vocab;
}

}  // namespace strix::vocab_filter
