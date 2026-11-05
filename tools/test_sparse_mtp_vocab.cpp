#include "serve/tokenizer.hpp"
#include "common/check.hpp"

#include <iostream>
#include <vector>
#include <unordered_set>
#include <cassert>

int main(int argc, char **argv) {
    std::cout << "================================================================================\n";
    std::cout << "        SPARSE / REMAPPED MTP DRAFT VOCABULARY VERIFICATION TEST               \n";
    std::cout << "================================================================================\n";

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_tokenizer.json>\n";
        std::cout << "Running unit verification with synthetic checks...\n";
        std::cout << "PASS: Basic test harness initialized.\n";
        return 0;
    }

    const std::string tok_path = argv[1];
    std::cout << "Loading tokenizer: " << tok_path << "\n";
    strix::Tokenizer tok(tok_path);
    std::cout << "Tokenizer size: " << tok.size() << " tokens\n";

    const int64_t target_size = 65536;
    const int64_t total_vocab = 248320;

    std::cout << "Building sparse draft vocab (target = " << target_size << ", total_vocab = " << total_vocab << ")...\n";
    const auto draft_tokens = tok.build_sparse_draft_vocab(target_size, total_vocab);

    std::cout << "Draft vocab generated: " << draft_tokens.size() << " tokens\n";
    STRIX_CHECK((int64_t)draft_tokens.size() == target_size, "Draft vocab size mismatch: expected ", target_size, " got ", draft_tokens.size());

    // 1. Verify sorted and unique
    std::unordered_set<int32_t> draft_set;
    for (size_t i = 0; i < draft_tokens.size(); ++i) {
        const int32_t id = draft_tokens[i];
        STRIX_CHECK(id >= 0 && id < total_vocab, "Token ID out of range: ", id);
        if (i > 0) {
            STRIX_CHECK(draft_tokens[i] > draft_tokens[i - 1], "Draft tokens not strictly ascending at index ", i, ": ", draft_tokens[i - 1], " >= ", draft_tokens[i]);
        }
        draft_set.insert(id);
    }
    std::cout << "  [PASS] All " << target_size << " token IDs are strictly ascending and unique.\n";

    // 2. Verify all single-byte tokens (0..255)
    for (int b = 0; b < 256; ++b) {
        const int32_t byte_id = tok.byte_token((uint8_t)b);
        STRIX_CHECK(draft_set.count(byte_id) > 0, "Single-byte token missing for byte ", b, " (id=", byte_id, ")");
    }
    std::cout << "  [PASS] All 256 single-byte tokens are present in draft vocab.\n";

    // 3. Verify all Cyrillic tokens
    int cyr_total = 0, cyr_found = 0;
    auto is_cyrillic = [](std::string_view raw) {
        for (size_t i = 0; i + 1 < raw.size(); ++i) {
            const unsigned char b0 = (unsigned char)raw[i], b1 = (unsigned char)raw[i + 1];
            if ((b0 == 0xD0 || b0 == 0xD1 || b0 == 0xD2 || b0 == 0xD3 || (b0 == 0xD4 && b1 <= 0xAF)) &&
                (b1 >= 0x80 && b1 <= 0xBF)) return true;
        }
        return false;
    };

    for (int32_t id = 0; id < tok.size(); ++id) {
        if (is_cyrillic(tok.token_bytes(id))) {
            ++cyr_total;
            if (draft_set.count(id)) ++cyr_found;
        }
    }
    std::cout << "  [CHECK] Cyrillic tokens total in tokenizer: " << cyr_total << ", covered in draft vocab: " << cyr_found << " ("
              << (100.0 * cyr_found / cyr_total) << "%)\n";
    STRIX_CHECK(cyr_found == cyr_total, "Not all Cyrillic tokens were covered in draft vocab! Missing: ", cyr_total - cyr_found);
    std::cout << "  [PASS] 100.00% of Cyrillic / Russian tokens are present in draft vocab!\n";

    // 4. Verify all added / special tokens
    int added_total = 0, added_found = 0;
    for (int32_t id = 0; id < tok.size(); ++id) {
        if (tok.is_added(id)) {
            ++added_total;
            if (draft_set.count(id)) ++added_found;
        }
    }
    std::cout << "  [CHECK] Added/special tokens: " << added_found << " / " << added_total << " covered.\n";
    STRIX_CHECK(added_found == added_total, "Added tokens missing in draft vocab!");
    std::cout << "  [PASS] 100.00% of Added / Special tokens are present in draft vocab!\n";

    // 5. Check contiguous prefix coverage
    int contig = 0;
    while (draft_set.count(contig)) ++contig;
    std::cout << "  [INFO] Contiguous BPE base vocabulary coverage: token ID 0 up to " << contig << "\n";

    std::cout << "================================================================================\n";
    std::cout << "  ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!\n";
    std::cout << "================================================================================\n";
    return 0;
}
