#include "runtime/page_pool.hpp"

#include <numeric>

namespace strix {

namespace {
size_t act_bytes(kernels::Act act) {
    switch (act) {
        case kernels::Act::BF16: return 2;
        case kernels::Act::F32:  return 4;
        default: return 2;
    }
}
}  // namespace

BlockManager::BlockManager(int64_t total_blocks)
    : total_blocks_(total_blocks), ref_counts_(total_blocks, 0) {
    STRIX_CHECK(total_blocks > 0, "BlockManager: total_blocks must be > 0 (got ", total_blocks, ")");
    free_list_.reserve((size_t)total_blocks);
    // Push blocks in reverse order so allocating pops from front (0, 1, 2...)
    for (int64_t i = total_blocks - 1; i >= 0; --i) {
        free_list_.push_back(static_cast<int32_t>(i));
    }
}

int64_t BlockManager::free_blocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int64_t>(free_list_.size());
}

int64_t BlockManager::used_blocks() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_blocks_ - static_cast<int64_t>(free_list_.size());
}

int32_t BlockManager::allocate_block() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_list_.empty()) return -1;
    int32_t b = free_list_.back();
    free_list_.pop_back();
    ref_counts_[(size_t)b] = 1;
    return b;
}

bool BlockManager::allocate_blocks(int64_t count, std::vector<int32_t> &out) {
    if (count <= 0) return true;
    std::lock_guard<std::mutex> lock(mutex_);
    if (static_cast<int64_t>(free_list_.size()) < count) return false;
    out.reserve(out.size() + (size_t)count);
    for (int64_t i = 0; i < count; ++i) {
        int32_t b = free_list_.back();
        free_list_.pop_back();
        ref_counts_[(size_t)b] = 1;
        out.push_back(b);
    }
    return true;
}

void BlockManager::free_block(int32_t block_id) {
    release_block(block_id);
}

void BlockManager::free_blocks(const int32_t *blocks, size_t count) {
    if (!blocks || count == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < count; ++i) {
        int32_t b = blocks[i];
        if (b < 0 || b >= total_blocks_) continue;
        if (ref_counts_[(size_t)b] <= 0) continue;
        if (--ref_counts_[(size_t)b] == 0) {
            free_list_.push_back(b);
        }
    }
}

void BlockManager::acquire_block(int32_t block_id) {
    STRIX_CHECK(block_id >= 0 && block_id < total_blocks_, "BlockManager::acquire_block: invalid id ", block_id);
    std::lock_guard<std::mutex> lock(mutex_);
    ++ref_counts_[(size_t)block_id];
}

void BlockManager::release_block(int32_t block_id) {
    if (block_id < 0 || block_id >= total_blocks_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (ref_counts_[(size_t)block_id] <= 0) return;
    if (--ref_counts_[(size_t)block_id] == 0) {
        free_list_.push_back(block_id);
    }
}

PagePool::PagePool(int64_t total_tokens, int64_t W, const std::vector<bool> &is_attention,
                   kernels::Act act, bool has_mtp)
    : total_blocks_((total_tokens + kPageBlockSize - 1) / kPageBlockSize),
      W_(W),
      act_(act),
      has_mtp_(has_mtp),
      block_mgr_(total_blocks_) {
    const size_t e = act_bytes(act);
    const size_t layer_bytes = (size_t)(total_blocks_ * kPageBlockSize * W_) * e;

    for (size_t i = 0; i < is_attention.size(); ++i) {
        const std::string tag = "layer " + std::to_string(i);
        if (is_attention[i]) {
            pool_k_.emplace_back(layer_bytes, "Paged K pool " + tag);
            pool_v_.emplace_back(layer_bytes, "Paged V pool " + tag);
        } else {
            pool_k_.emplace_back();
            pool_v_.emplace_back();
        }
    }

    if (has_mtp) {
        pool_k_mtp_ = DeviceBuffer<uint8_t>(layer_bytes, "Paged MTP K pool");
        pool_v_mtp_ = DeviceBuffer<uint8_t>(layer_bytes, "Paged MTP V pool");
    }
}

}  // namespace strix
