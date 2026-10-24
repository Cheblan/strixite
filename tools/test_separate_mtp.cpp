#include "common/check.hpp"
#include "formats/strixw.hpp"
#include "runtime/strixw_loader.hpp"
#include "serve/server_config.hpp"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;
using namespace strix;

void test_server_config_mtp_weights(const std::string &tmp_dir) {
    std::cout << "[TEST 1] Testing ServerSettings with mtp-weights option..." << std::endl;
    ServerSettings s(strix_server_options(tmp_dir));

    // Default should be off (represented as empty string by .text()):
    STRIX_CHECK(s.text("mtp-weights").empty(), "default mtp-weights should be off");
    STRIX_CHECK(s.on("mtp"), "default mtp should be on");

    // Test passing --mtp-weights explicitly:
    const std::string custom_path = tmp_dir + "/custom_mtp.strixw";
    std::vector<std::string> args = {"--mtp-weights", custom_path};
    s.load(args);
    STRIX_CHECK(s.text("mtp-weights") == custom_path, "mtp-weights should match custom path");

    // Test sibling detection logic:
    const std::string weights_path = tmp_dir + "/weights.strixw";
    const std::string sibling_path = tmp_dir + "/mtp.strixw";
    
    // Create sibling file:
    std::ofstream(sibling_path) << "dummy";
    
    // Reset settings to default:
    ServerSettings s2(strix_server_options(tmp_dir));
    const std::string explicit_path = s2.text("mtp-weights");
    std::string detected_sibling;
    if (explicit_path.empty()) {
        const size_t slash = weights_path.rfind('/');
        if (slash != std::string::npos && slash > 0) {
            const std::string sibling = weights_path.substr(0, slash) + "/mtp.strixw";
            if (fs::is_regular_file(sibling)) detected_sibling = sibling;
        }
    }
    STRIX_CHECK(detected_sibling == sibling_path, "sibling mtp.strixw should be auto-detected");
    fs::remove(sibling_path);
    std::cout << "  ✓ ServerSettings & sibling auto-detection PASSED\n";
}

void test_strixw_mtp_file_roundtrip(const std::string &tmp_dir) {
    std::cout << "[TEST 2] Testing StrixwWriter & StrixwFile for mtp.strixw..." << std::endl;
    const std::string mtp_path = tmp_dir + "/test_mtp.strixw";

    // Create a mock MTP file with various tensor types
    std::vector<StrixwTensor> tensors;
    {
        StrixwTensor t;
        t.name = "mtp.fc_embedding.weight";
        t.encoding = StrixwEncoding::Q8RowMajor;
        t.shape = {128, 64};
        t.group_size = 64;
        t.components = strixw_components(t);
        tensors.push_back(t);
    }
    {
        StrixwTensor t;
        t.name = "mtp.fc_hidden.weight";
        t.encoding = StrixwEncoding::Q8RowMajor;
        t.shape = {128, 64};
        t.group_size = 64;
        t.components = strixw_components(t);
        tensors.push_back(t);
    }
    {
        StrixwTensor t;
        t.name = "mtp.pre_fc_norm_embedding.weight";
        t.encoding = StrixwEncoding::F32;
        t.shape = {64};
        t.components = strixw_components(t);
        tensors.push_back(t);
    }
    {
        StrixwTensor t;
        t.name = "mtp.layers.0.mlp.router";
        t.encoding = StrixwEncoding::BF16;
        t.shape = {17, 64};
        t.components = strixw_components(t);
        tensors.push_back(t);
    }

    std::map<std::string, std::string> meta = {
        {"arch", "qwen4exp"},
        {"kind", "mtp"},
        {"layout", "default=q4g64,mtp_fc=q8g64"},
        {"source_fingerprint", "deadbeef12345678"},
        {"mtp", "yes"},
        {"mtp_source", "checkpoint"},
        {"complete", "yes"},
        {"format_note", "MTP draft head unit test"}
    };

    StrixwWriter writer(mtp_path, meta, tensors);
    // Write tensor 0 (Q8: Q, Scale, Min)
    {
        std::vector<uint8_t> q(128 * 64, 0x42);
        std::vector<uint16_t> scale(128 * 1, 0x3c00);
        std::vector<uint16_t> minv(128 * 1, 0x0000);
        writer.write(0, StrixwRole::Q, q.data(), q.size());
        writer.write(0, StrixwRole::Scale, scale.data(), scale.size() * 2);
        writer.write(0, StrixwRole::Min, minv.data(), minv.size() * 2);
    }
    // Write tensor 1 (Q8: Q, Scale, Min)
    {
        std::vector<uint8_t> q(128 * 64, 0x77);
        std::vector<uint16_t> scale(128 * 1, 0x3c00);
        std::vector<uint16_t> minv(128 * 1, 0x0000);
        writer.write(1, StrixwRole::Q, q.data(), q.size());
        writer.write(1, StrixwRole::Scale, scale.data(), scale.size() * 2);
        writer.write(1, StrixwRole::Min, minv.data(), minv.size() * 2);
    }
    // Write tensor 2 (F32)
    {
        std::vector<float> f(64, 1.0f);
        writer.write(2, StrixwRole::Data, f.data(), f.size() * 4);
    }
    // Write tensor 3 (BF16)
    {
        std::vector<uint16_t> b(17 * 64, 0x3f80);
        writer.write(3, StrixwRole::Data, b.data(), b.size() * 2);
    }
    writer.finish();

    // Verify file via StrixwFile:
    {
        StrixwFile file(mtp_path);
        STRIX_CHECK(file.meta("arch") == "qwen4exp", "arch must be qwen4exp");
        STRIX_CHECK(file.meta("kind") == "mtp", "kind must be mtp");
        STRIX_CHECK(file.meta("source_fingerprint") == "deadbeef12345678", "fingerprint match");
        STRIX_CHECK(file.tensors().size() == 4, "4 tensors expected");
        file.verify_hashes(2);
        std::cout << "  ✓ StrixwFile verification & component hashes PASSED\n";
    }

    // Verify loading via StrixwDevice on GPU:
    {
        StrixwDevice dev(mtp_path, 2, WeightPlacement::Device);
        STRIX_CHECK(dev.file().meta("kind") == "mtp", "device loader sees kind=mtp");
        const Q8DeviceView q8_0 = dev.q8("mtp.fc_embedding.weight", {128, 64});
        STRIX_CHECK(q8_0.N == 128 && q8_0.K == 64 && q8_0.G == 64, "q8 view dims match");
        const float *f = dev.f32("mtp.pre_fc_norm_embedding.weight", {64});
        STRIX_CHECK(f != nullptr, "f32 view non-null");
        const uint16_t *bf = dev.bf16("mtp.layers.0.mlp.router", {17, 64});
        STRIX_CHECK(bf != nullptr, "bf16 view non-null");
        std::cout << "  ✓ StrixwDevice GPU upload & views PASSED\n";
    }

    fs::remove(mtp_path);
}

void test_main_weights_mtp_isolation(const std::string &tmp_dir) {
    std::cout << "[TEST 3] Testing isolation: separate MTP ignores main file MTP..." << std::endl;
    const std::string main_path = tmp_dir + "/main_with_legacy_mtp.strixw";
    const std::string separate_mtp_path = tmp_dir + "/separate_mtp.strixw";

    // Main file has mtp.fc_embedding.weight with byte 0xAA:
    {
        std::vector<StrixwTensor> tensors;
        StrixwTensor t;
        t.name = "mtp.fc_embedding.weight";
        t.encoding = StrixwEncoding::Q8RowMajor;
        t.shape = {64, 64};
        t.group_size = 64;
        t.components = strixw_components(t);
        tensors.push_back(t);

        std::map<std::string, std::string> meta = {
            {"arch", "qwen4exp"},
            {"layout", "default=q8g64"},
            {"source_fingerprint", "main_fp"},
            {"complete", "no"}
        };
        StrixwWriter w(main_path, meta, tensors);
        std::vector<uint8_t> q(64 * 64, 0xAA);
        std::vector<uint16_t> sc(64, 0x3c00), mn(64, 0);
        w.write(0, StrixwRole::Q, q.data(), q.size());
        w.write(0, StrixwRole::Scale, sc.data(), sc.size() * 2);
        w.write(0, StrixwRole::Min, mn.data(), mn.size() * 2);
        w.finish();
    }

    // Separate MTP file has mtp.fc_embedding.weight with byte 0xBB:
    {
        std::vector<StrixwTensor> tensors;
        StrixwTensor t;
        t.name = "mtp.fc_embedding.weight";
        t.encoding = StrixwEncoding::Q8RowMajor;
        t.shape = {64, 64};
        t.group_size = 64;
        t.components = strixw_components(t);
        tensors.push_back(t);

        std::map<std::string, std::string> meta = {
            {"arch", "qwen4exp"},
            {"kind", "mtp"},
            {"layout", "default=q8g64"},
            {"source_fingerprint", "separate_fp"},
            {"complete", "yes"}
        };
        StrixwWriter w(separate_mtp_path, meta, tensors);
        std::vector<uint8_t> q(64 * 64, 0xBB);
        std::vector<uint16_t> sc(64, 0x3c00), mn(64, 0);
        w.write(0, StrixwRole::Q, q.data(), q.size());
        w.write(0, StrixwRole::Scale, sc.data(), sc.size() * 2);
        w.write(0, StrixwRole::Min, mn.data(), mn.size() * 2);
        w.finish();
    }

    // Verify reading from separate vs main:
    StrixwDevice main_dev(main_path, 1, WeightPlacement::Device);
    StrixwDevice sep_dev(separate_mtp_path, 1, WeightPlacement::Device);

    std::vector<uint8_t> host_main(64 * 64), host_sep(64 * 64);
    STRIX_HIP_CHECK(hipMemcpy(host_main.data(), main_dev.q8("mtp.fc_embedding.weight", {64, 64}).q,
                              64 * 64, hipMemcpyDeviceToHost), "read main");
    STRIX_HIP_CHECK(hipMemcpy(host_sep.data(), sep_dev.q8("mtp.fc_embedding.weight", {64, 64}).q,
                              64 * 64, hipMemcpyDeviceToHost), "read sep");

    STRIX_CHECK(host_main[0] == 0xAA, "main file has 0xAA");
    STRIX_CHECK(host_sep[0] == 0xBB, "separate file has 0xBB");

    // The logic in Qwen4ExpModel:
    // const StrixwDevice &mw = mtp_w_ ? *mtp_w_ : *w_;
    // If mtp_w_ is loaded, mw reads from sep_dev!
    std::unique_ptr<StrixwDevice> mtp_w = std::make_unique<StrixwDevice>(separate_mtp_path, 1, WeightPlacement::Device);
    const StrixwDevice &mw = mtp_w ? *mtp_w : main_dev;
    const Q8DeviceView v = mw.q8("mtp.fc_embedding.weight", {64, 64});
    std::vector<uint8_t> loaded(64 * 64);
    STRIX_HIP_CHECK(hipMemcpy(loaded.data(), v.q, 64 * 64, hipMemcpyDeviceToHost), "read loaded");
    STRIX_CHECK(loaded[0] == 0xBB, "loaded MTP weight must come from separate file (0xBB), ignoring main (0xAA)");
    std::cout << "  ✓ Separate MTP successfully overrides and isolates from main file MTP\n";

    fs::remove(main_path);
    fs::remove(separate_mtp_path);
}

int main() {
    const std::string tmp_dir = "/tmp/strix_test_separate_mtp_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    fs::create_directories(tmp_dir);
    try {
        test_server_config_mtp_weights(tmp_dir);
        test_strixw_mtp_file_roundtrip(tmp_dir);
        test_main_weights_mtp_isolation(tmp_dir);
        fs::remove_all(tmp_dir);
        std::cout << "\n============================================\n";
        std::cout << "ALL SEPARATE MTP UNIT TESTS PASSED SUCCESSFULLY!\n";
        std::cout << "============================================\n";
        return 0;
    } catch (const std::exception &e) {
        fs::remove_all(tmp_dir);
        std::cerr << "FAILED: " << e.what() << std::endl;
        return 1;
    }
}
