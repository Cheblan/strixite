#pragma once

// A device array that takes memory as it is used: the address range for the
// full capacity is reserved once (hipMemAddressReserve), and physical pages (hipMemCreate) are mapped into it where
// it is used - kernels see one array at a fixed pointer with its layout unchanged, as with DeviceBuffer. An array of
// several capacity-sized regions (the 8-bit K/V: codes and params of K and V) maps each region's used prefix: ranges
// of bytes, any page they touch. On Strix Halo the GPU's pages come from system RAM, so a slot's K/V costs the
// deepest point its conversations have reached, not its full capacity.
//
// A mapped page stays mapped until the buffer is destroyed - there is no runtime release. On gfx1151, after a page
// was unmapped and released and a new one mapped at the same address, kernels sometimes read memory other than the
// new page there (another allocation's, once the freed memory was reused), with nothing failing: the same forward
// gave different logits, only while other processes used the GPU.
//
// Pages are only ever mapped back into this buffer's own reservation: on gfx1151 a page mapped into a second
// reservation reads the wrong memory.
//
// Not thread-safe (as DeviceBuffer). Growing while a kernel reads the range is safe: ensure() maps pages that weren't
// mapped, so kernels reading mapped bytes are unaffected.

#include "common/check.hpp"
#include "common/hip_check.hpp"
#include "runtime/device_buffer.hpp"

#include "common/hip_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strix {

class PagedDeviceBuffer {
public:
    PagedDeviceBuffer() = default;
    // capacity_bytes: the most this array will ever hold (rounded up to whole pages); page_bytes: the mapping unit, a
    // multiple of the device's minimum VMM granularity. No page is mapped yet: get() is valid, size() is 0.
    PagedDeviceBuffer(size_t capacity_bytes, size_t page_bytes, std::string name) : name_(std::move(name)) {
        STRIX_CHECK(capacity_bytes > 0, "paged buffer '", name_, "': capacity of 0 bytes");
        STRIX_CHECK(page_bytes > 0, "paged buffer '", name_, "': page of 0 bytes");
        STRIX_HIP_CHECK(hipGetDevice(&dev_), "paged buffer '", name_, "': current device");
        int vmm = 0;
        STRIX_HIP_CHECK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, dev_),
                        "paged buffer '", name_, "': VMM support attribute");
        STRIX_CHECK(vmm == 1, "paged buffer '", name_, "': device ", dev_, " reports no virtual memory management ",
                    "(attribute ", vmm, ")");
        const hipMemAllocationProp prop = page_prop();
        size_t gran = 0;
        STRIX_HIP_CHECK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum),
                        "paged buffer '", name_, "': granularity");
        STRIX_CHECK(gran > 0 && page_bytes % gran == 0, "paged buffer '", name_, "': page of ", page_bytes,
                    " B is not a multiple of the device's VMM granularity ", gran, " B");
        STRIX_CHECK(capacity_bytes <= SIZE_MAX - page_bytes, "paged buffer '", name_, "': capacity ", capacity_bytes,
                    " B overflows when rounded to ", page_bytes, " B pages");
        page_ = page_bytes;
        reserved_ = (capacity_bytes + page_bytes - 1) / page_bytes * page_bytes;
        STRIX_HIP_CHECK(hipMemAddressReserve(&base_, reserved_, gran, nullptr, 0), "paged buffer '", name_,
                        "': reserving ", reserved_, " B of address space");
        pages_.assign(reserved_ / page_, nullptr);
    }
    ~PagedDeviceBuffer() { release(); }
    PagedDeviceBuffer(PagedDeviceBuffer &&o) noexcept { swap(o); }
    PagedDeviceBuffer &operator=(PagedDeviceBuffer &&o) noexcept {
        if (this != &o) {
            release();
            swap(o);
        }
        return *this;
    }
    PagedDeviceBuffer(const PagedDeviceBuffer &) = delete;
    PagedDeviceBuffer &operator=(const PagedDeviceBuffer &) = delete;

    // Bytes [off, off + bytes) of the array (bytes 0: none).
    struct Range {
        size_t off = 0, bytes = 0;
    };

    uint8_t *get() const { return static_cast<uint8_t *>(base_); }
    size_t capacity() const { return reserved_; }  // bytes the range can ever map
    size_t size() const { return mapped_ * page_; }  // bytes mapped now (whole pages)
    size_t page_bytes() const { return page_; }
    // The bytes ensure(ranges) would map (pages they touch that aren't mapped): what the caller asks the memory budget
    // for before growing.
    size_t missing_bytes(const std::vector<Range> &ranges) const {
        std::vector<char> seen(pages_.size(), 0);  // ranges may share a page: count it once
        size_t n = 0;
        for_pages(ranges, "missing_bytes", [&](size_t p) {
            if (!seen[p] && pages_[p] == nullptr) seen[p] = 1, n += page_;
        });
        return n;
    }
    size_t grow_bytes(size_t bytes) const { return missing_bytes({{0, bytes}}); }
    // The bytes of every page the ranges touch, mapped or not (what they cost when fully mapped).
    size_t touched_bytes(const std::vector<Range> &ranges) const {
        std::vector<char> seen(pages_.size(), 0);
        size_t n = 0;
        for_pages(ranges, "touched_bytes", [&](size_t p) {
            if (!seen[p]) seen[p] = 1, n += page_;
        });
        return n;
    }
    // Maps every page the ranges touch. Mapped pages keep their contents; new pages' contents are undefined (the
    // caller writes before it reads, as with a fresh DeviceBuffer).
    void ensure(const std::vector<Range> &ranges) {
        for_pages(ranges, "ensure", [&](size_t p) {
            if (pages_[p] == nullptr) map_page(p);
        });
    }
    void ensure(size_t bytes) { ensure({{0, bytes}}); }

private:
    hipMemAllocationProp page_prop() const {
        hipMemAllocationProp p = {};
        p.type = hipMemAllocationTypePinned;
        p.location.type = hipMemLocationTypeDevice;
        p.location.id = dev_;
        return p;
    }
    template <typename F>
    void for_pages(const std::vector<Range> &ranges, const char *what, F &&f) const {
        STRIX_CHECK(base_ != nullptr, "paged buffer '", name_, "': ", what, " on an empty (moved-from or default) buffer");
        for (const Range &r : ranges) {
            STRIX_CHECK(r.off <= reserved_ && r.bytes <= reserved_ - r.off, "paged buffer '", name_, "': ", what, " of [",
                        r.off, ", +", r.bytes, " B) past its capacity of ", reserved_, " B");
            if (r.bytes == 0) continue;
            for (size_t p = r.off / page_; p <= (r.off + r.bytes - 1) / page_; ++p) f(p);
        }
    }
    void map_page(size_t p) {
        hipMemAccessDesc access = {};
        access.location.type = hipMemLocationTypeDevice;
        access.location.id = dev_;
        access.flags = hipMemAccessFlagsProtReadWrite;
        const hipMemAllocationProp prop = page_prop();
        char *at = static_cast<char *>(base_) + p * page_;
        hipMemGenericAllocationHandle_t h{};
        STRIX_HIP_CHECK(hipMemCreate(&h, page_, &prop, 0), "paged buffer '", name_, "': creating page ", p, " (", page_,
                        " B; ", size(), " of ", reserved_, " B mapped)");
        const hipError_t m = hipMemMap(at, page_, 0, h, 0);
        if (m != hipSuccess) {
            (void)hipMemRelease(h);
            STRIX_HIP_CHECK(m, "paged buffer '", name_, "': mapping page ", p);
        }
        const hipError_t a = hipMemSetAccess(at, page_, &access, 1);
        if (a != hipSuccess) {
            (void)hipMemUnmap(at, page_);
            (void)hipMemRelease(h);
            STRIX_HIP_CHECK(a, "paged buffer '", name_, "': device access for page ", p);
        }
        pages_[p] = h;
        ++mapped_;
        GpuMemoryTally::add(name_, (int64_t)page_);
    }
    void release() noexcept {  // destructor path: no throwing; a failed unmap leaks rather than aborts
        if (base_ == nullptr) return;
        for (size_t p = 0; p < pages_.size(); ++p) {
            if (pages_[p] == nullptr) continue;
            if (hipMemUnmap(static_cast<char *>(base_) + p * page_, page_) == hipSuccess) (void)hipMemRelease(pages_[p]);
            GpuMemoryTally::add(name_, -(int64_t)page_);
        }
        pages_.clear();
        mapped_ = 0;
        (void)hipMemAddressFree(base_, reserved_);
        base_ = nullptr;
        reserved_ = 0;
    }
    void swap(PagedDeviceBuffer &o) noexcept {
        std::swap(base_, o.base_);
        std::swap(reserved_, o.reserved_);
        std::swap(page_, o.page_);
        std::swap(dev_, o.dev_);
        std::swap(pages_, o.pages_);
        std::swap(mapped_, o.mapped_);
        std::swap(name_, o.name_);
    }

    void *base_ = nullptr;
    size_t reserved_ = 0, page_ = 0, mapped_ = 0;
    int dev_ = 0;
    std::vector<hipMemGenericAllocationHandle_t> pages_;  // one a page of the range; null: unmapped
    std::string name_;
};

}  // namespace strix
