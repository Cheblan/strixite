#pragma once

#include "runtime/device_buffer.hpp"
#include "kernels/norm.hpp"
#include "common/check.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace strix {

// Number of tokens per physical block.
// Must be 32 so that:
// 1) QSA micro-blocks (4 tokens) fit evenly (8 micro-blocks per page).
// 2) Kv8 quantization groups (32 tokens) align 1:1 with the page.
constexpr int64_t kPageBlockSize = 32;

// Thread-safe manager of physical page block IDs.
class BlockManager {
public:
    explicit BlockManager(int64_t total_blocks);

    int64_t total_blocks() const { return total_blocks_; }
    int64_t free_blocks() const;
    int64_t used_blocks() const;

    // Allocate 1 physical block. Returns -1 if exhausted.
    int32_t allocate_block();

    // Allocate count physical blocks. Returns true on success (appending to out), false if not enough.
    bool allocate_blocks(int64_t count, std::vector<int32_t> &out);

    // Free a previously allocated block.
    void free_block(int32_t block_id);

    // Free multiple blocks.
    void free_blocks(const int32_t *blocks, size_t count);

    // Copy-on-Write reference counting:
    void acquire_block(int32_t block_id);
    void release_block(int32_t block_id);

private:
    const int64_t total_blocks_;
    mutable std::mutex mutex_;
    std::vector<int32_t> free_list_;
    std::vector<int32_t> ref_counts_;
};

// Physical device memory pool for K and V cache blocks across all attention layers.
class PagePool {
public:
    PagePool(int64_t total_tokens, int64_t W, const std::vector<bool> &is_attention,
             kernels::Act act = kernels::Act::BF16, bool has_mtp = false);

    int64_t total_tokens() const { return total_blocks_ * kPageBlockSize; }
    int64_t total_blocks() const { return total_blocks_; }
    int64_t block_size() const { return kPageBlockSize; }
    int64_t W() const { return W_; }
    kernels::Act act() const { return act_; }
    bool has_mtp() const { return has_mtp_; }

    BlockManager &block_mgr() { return block_mgr_; }
    const BlockManager &block_mgr() const { return block_mgr_; }

    // Physical memory buffers per attention layer (or nullptr for non-attention layers)
    uint8_t *k_pool(size_t layer) { return layer < pool_k_.size() && pool_k_[layer].size() ? pool_k_[layer].get() : nullptr; }
    const uint8_t *k_pool(size_t layer) const { return layer < pool_k_.size() && pool_k_[layer].size() ? pool_k_[layer].get() : nullptr; }
    uint8_t *v_pool(size_t layer) { return layer < pool_v_.size() && pool_v_[layer].size() ? pool_v_[layer].get() : nullptr; }
    const uint8_t *v_pool(size_t layer) const { return layer < pool_v_.size() && pool_v_[layer].size() ? pool_v_[layer].get() : nullptr; }

    uint8_t *k_mtp_pool() { return pool_k_mtp_.size() ? pool_k_mtp_.get() : nullptr; }
    const uint8_t *k_mtp_pool() const { return pool_k_mtp_.size() ? pool_k_mtp_.get() : nullptr; }
    uint8_t *v_mtp_pool() { return pool_v_mtp_.size() ? pool_v_mtp_.get() : nullptr; }
    const uint8_t *v_mtp_pool() const { return pool_v_mtp_.size() ? pool_v_mtp_.get() : nullptr; }

private:
    const int64_t total_blocks_;
    const int64_t W_;
    const kernels::Act act_;
    const bool has_mtp_;
    BlockManager block_mgr_;
    std::vector<DeviceBuffer<uint8_t>> pool_k_;
    std::vector<DeviceBuffer<uint8_t>> pool_v_;
    DeviceBuffer<uint8_t> pool_k_mtp_;
    DeviceBuffer<uint8_t> pool_v_mtp_;
};

}  // namespace strix
