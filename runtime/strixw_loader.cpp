#include "runtime/strixw_loader.hpp"

#include "common/check.hpp"
#include "common/hip_check.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

namespace strix {

namespace {

constexpr size_t kStageBytes = 256ull << 20;

std::string shape_str(const std::vector<int64_t> &s) {
    std::string r = "[";
    for (size_t i = 0; i < s.size(); ++i) r += (i ? ", " : "") + std::to_string(s[i]);
    return r + "]";
}

// n bytes at file offset off into dst, split over `threads` concurrent preads (NVMe queue depth).
void pread_parallel(int fd, void *dst, size_t n, uint64_t off, int threads, const std::string &path) {
    std::vector<std::thread> pool;
    std::vector<std::string> errs((size_t)threads);
    const size_t per = (n + (size_t)threads - 1) / (size_t)threads;
    for (int t = 0; t < threads; ++t)
        pool.emplace_back([&, t] {
            size_t a = std::min(n, (size_t)t * per), b = std::min(n, a + per);
            auto *p = static_cast<uint8_t *>(dst) + a;
            while (a < b) {
                const ssize_t r = ::pread(fd, p, b - a, (off_t)(off + a));
                if (r < 0 && errno == EINTR) continue;
                if (r <= 0) {
                    errs[(size_t)t] = r < 0 ? std::strerror(errno) : "unexpected end of file";
                    return;
                }
                a += (size_t)r, p += r;
            }
        });
    for (std::thread &th : pool) th.join();
    for (const std::string &e : errs)
        STRIX_CHECK(e.empty(), "StrixwDevice: reading '", path, "' at ", off, " (", n, " bytes) failed: ", e);
}

double mem_available_gib() {
    std::ifstream f("/proc/meminfo");
    STRIX_CHECK(f.good(), "StrixwDevice: can't open /proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) != 0) continue;
        long long kb = -1;
        std::istringstream(line.substr(13)) >> kb;
        STRIX_CHECK(kb >= 0, "StrixwDevice: unparsable /proc/meminfo line '", line, "'");
        return (double)kb / (1024.0 * 1024.0);
    }
    STRIX_FAIL("StrixwDevice: no MemAvailable line in /proc/meminfo");
}

}  // namespace

const char *weight_placement_name(WeightPlacement p) {
    switch (p) {
        case WeightPlacement::Device: return "device";
        case WeightPlacement::HostExperts: return "host-experts";
    }
    return "?";
}

WeightPlacement parse_weight_placement(const std::string &v, const char *what) {
    if (v == "device") return WeightPlacement::Device;
    if (v == "host-experts") return WeightPlacement::HostExperts;
    STRIX_FAIL(what, " is '", v, "', expected device or host-experts");
}

bool is_host_expert_tensor(const std::string &name) {
    return name.find("mlp.experts_gate_up") != std::string::npos || name.find("mlp.experts_down") != std::string::npos;
}

StrixwDevice::StrixwDevice(const std::string &path, int read_threads, WeightPlacement placement,
                           double host_margin_gib)
    : placement_(placement) {
    STRIX_CHECK(read_threads >= 1 && read_threads <= 64, "StrixwDevice: read_threads = ", read_threads);
    STRIX_CHECK(placement == WeightPlacement::Device || placement == WeightPlacement::HostExperts,
                "StrixwDevice: unknown placement ", (int)placement);
    STRIX_CHECK(std::isfinite(host_margin_gib) && host_margin_gib >= 0, "StrixwDevice: host_margin_gib = ",
                host_margin_gib, ", expected a finite value >= 0");
    const auto t0 = std::chrono::steady_clock::now();
    file_ = std::make_unique<StrixwFile>(path);
    const uint64_t off = file_->data_offset(), n = file_->file_bytes() - off;
    STRIX_CHECK(n > 0, "StrixwDevice: '", path, "' has an empty data region");

    // Runs of consecutive tensors with one placement; a run owns the bytes up to the next run (padding included).
    std::vector<std::pair<uint64_t, bool>> starts;  // tensor start, host
    for (const StrixwTensor &t : file_->tensors()) {
        STRIX_CHECK(!t.components.empty(), "StrixwDevice: '", t.name, "' in '", path, "' has no components");
        uint64_t s = UINT64_MAX;
        for (const StrixwComponent &c : t.components) s = std::min(s, c.offset);
        STRIX_CHECK(s >= off && s < off + n, "StrixwDevice: '", t.name, "' starts at ", s, ", outside the data region [",
                    off, ", ", off + n, ")");
        starts.emplace_back(s, placement == WeightPlacement::HostExperts && is_host_expert_tensor(t.name));
    }
    std::sort(starts.begin(), starts.end());
    for (const auto &[s, host] : starts) {
        if (runs_.empty()) runs_.push_back({off, 0, 0, host});
        else if (runs_.back().host != host) runs_.back().end = s, runs_.push_back({s, 0, 0, host});
    }
    if (runs_.empty()) runs_.push_back({off, 0, 0, false});
    runs_.back().end = off + n;
    // Each run at its region's next offset with the same remainder mod 4096 as its file offset.
    constexpr uint64_t kA = 4096;
    for (Run &r : runs_) {
        STRIX_CHECK(r.end > r.start, "StrixwDevice: empty run [", r.start, ", ", r.end, ") in '", path, "'");
        uint64_t &cursor = r.host ? host_bytes_ : device_bytes_;
        r.at = (cursor + kA - 1) / kA * kA + r.start % kA;
        cursor = r.at + (r.end - r.start);
    }
    STRIX_CHECK(placement == WeightPlacement::Device || host_bytes_ > 0, "StrixwDevice: host-experts placement, but '",
                path, "' has no mlp.experts_gate_up / mlp.experts_down tensors");

    if (device_bytes_ > 0) data_ = DeviceBuffer<uint8_t>((size_t)device_bytes_, "strixw data region of '" + path + "'");
    if (host_bytes_ > 0) {
        const double need = (double)host_bytes_ / 1073741824.0, avail = mem_available_gib();
        STRIX_CHECK(avail - need >= host_margin_gib, "StrixwDevice: host-experts placement of '", path, "' needs ",
                    need, " GiB pinned; MemAvailable is ", avail, " GiB, which would leave ", avail - need,
                    " GiB, below the ", host_margin_gib, " GiB margin");
        void *h = nullptr;
        STRIX_HIP_CHECK(hipHostMalloc(&h, (size_t)host_bytes_, hipHostMallocMapped | hipHostMallocNonCoherent),
                        "StrixwDevice: pinned host region for the experts of '", path, "', ", host_bytes_, " bytes");
        host_.reset(h);
        void *d = nullptr;
        STRIX_HIP_CHECK(hipHostGetDevicePointer(&d, host_.get(), 0), "StrixwDevice: device pointer of the host region");
        STRIX_CHECK(d != nullptr, "StrixwDevice: hipHostGetDevicePointer returned null for the host region");
        host_dev_ = static_cast<const uint8_t *>(d);
    }

    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    STRIX_CHECK(fd >= 0, "StrixwDevice: open '", path, "' failed: ", std::strerror(errno));
    void *stage[2] = {nullptr, nullptr};
    hipEvent_t done[2] = {nullptr, nullptr};
    hipStream_t stream = nullptr;
    try {
        STRIX_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "StrixwDevice: stream");
        for (int i = 0; i < 2; ++i) {
            STRIX_HIP_CHECK(hipHostMalloc(&stage[i], kStageBytes, hipHostMallocDefault), "StrixwDevice: staging ", i);
            STRIX_HIP_CHECK(hipEventCreateWithFlags(&done[i], hipEventDisableTiming), "StrixwDevice: event ", i);
        }
        bool pending[2] = {false, false};
        int cur = 0;
        for (const Run &r : runs_) {
            const uint64_t len_run = r.end - r.start;
            if (r.host) {
                // Straight into the pinned region; no kernel reads it before the constructor returns.
                pread_parallel(fd, static_cast<uint8_t *>(host_.get()) + r.at, (size_t)len_run, r.start, read_threads, path);
                continue;
            }
            for (uint64_t at = 0; at < len_run; at += kStageBytes, cur ^= 1) {
                const size_t len = (size_t)std::min<uint64_t>(kStageBytes, len_run - at);
                // Gate: this staging buffer's previous copy must be complete before it's overwritten.
                if (pending[cur]) STRIX_HIP_CHECK(hipEventSynchronize(done[cur]), "StrixwDevice: staging gate");
                pread_parallel(fd, stage[cur], len, r.start + at, read_threads, path);
                STRIX_HIP_CHECK(hipMemcpyAsync(data_.get() + r.at + at, stage[cur], len, hipMemcpyHostToDevice, stream),
                                "StrixwDevice: copy at ", r.start + at, " (", len, " bytes)");
                STRIX_HIP_CHECK(hipEventRecord(done[cur], stream), "StrixwDevice: event record");
                pending[cur] = true;
            }
        }
        STRIX_HIP_CHECK(hipStreamSynchronize(stream), "StrixwDevice: final sync");
    } catch (...) {
        for (int i = 0; i < 2; ++i) {
            if (done[i]) (void)hipEventDestroy(done[i]);
            if (stage[i]) (void)hipHostFree(stage[i]);
        }
        if (stream) (void)hipStreamDestroy(stream);
        ::close(fd);
        throw;
    }
    for (int i = 0; i < 2; ++i) (void)hipEventDestroy(done[i]), (void)hipHostFree(stage[i]);
    (void)hipStreamDestroy(stream);
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // the device / pinned copy is the one kept
    ::close(fd);
    load_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void StrixwDevice::HostFree::operator()(void *p) const {
    if (p) (void)hipHostFree(p);
}

const StrixwTensor &StrixwDevice::tensor(const std::string &name, StrixwEncoding enc,
                                         const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = file_->get(name);
    STRIX_CHECK(t.encoding == enc && t.shape == shape, "StrixwDevice: '", name, "' in '", file_->path(), "' is ",
                strixw_encoding_name(t.encoding), " ", shape_str(t.shape), ", expected ", strixw_encoding_name(enc),
                " ", shape_str(shape));
    return t;
}

const void *StrixwDevice::ptr(const StrixwTensor &t, StrixwRole role) const {
    for (const StrixwComponent &c : t.components)
        if (c.role == role) {
            // The run holding the component: the last one starting at or before it.
            auto it = std::upper_bound(runs_.begin(), runs_.end(), c.offset,
                                       [](uint64_t o, const Run &r) { return o < r.start; });
            STRIX_CHECK(it != runs_.begin(), "StrixwDevice: '", t.name, "' ", strixw_role_name(role), " at ",
                        c.offset, " is before the data region (", file_->data_offset(), ")");
            const Run &r = *(it - 1);
            STRIX_CHECK(c.offset + c.bytes <= r.end, "StrixwDevice: '", t.name, "' ", strixw_role_name(role), " [",
                        c.offset, ", ", c.offset + c.bytes, ") crosses its run's end ", r.end);
            return (r.host ? host_dev_ : data_.get()) + r.at + (c.offset - r.start);
        }
    STRIX_CHECK(false, "StrixwDevice: '", t.name, "' has no ", strixw_role_name(role), " component");
    return nullptr;
}

Q4DeviceView StrixwDevice::q4(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q4RowMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

Q8DeviceView StrixwDevice::q8(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q8RowMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

Q6DeviceView StrixwDevice::q6(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q6RowMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

Q5DeviceView StrixwDevice::q5(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q5RowMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

QWeightView StrixwDevice::qw(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = file_->get(name);
    switch (t.encoding) {
        case StrixwEncoding::Q4RowMajor: return qweight(q4(name, shape));
        case StrixwEncoding::Q8RowMajor: return qweight(q8(name, shape));
        case StrixwEncoding::Q6RowMajor: return qweight(q6(name, shape));
        case StrixwEncoding::Q5RowMajor: return qweight(q5(name, shape));
        default:
            STRIX_FAIL("StrixwDevice: '", name, "' in '", file_->path(), "' is ", strixw_encoding_name(t.encoding), " ",
                       shape_str(t.shape), ", expected a dense projection: q4, q5, q6 or q8 row-major ", shape_str(shape));
    }
}

Q4ChunkMajorView StrixwDevice::q4_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q4ChunkMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

Q8ChunkMajorView StrixwDevice::q8_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = tensor(name, StrixwEncoding::Q8ChunkMajor, shape);
    return {static_cast<const uint8_t *>(ptr(t, StrixwRole::Q)), static_cast<const uint16_t *>(ptr(t, StrixwRole::Scale)),
            static_cast<const uint16_t *>(ptr(t, StrixwRole::Min)), shape[0], shape[1], t.group_size};
}

QChunkMajorView StrixwDevice::qw_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const {
    const StrixwTensor &t = file_->get(name);
    switch (t.encoding) {
        case StrixwEncoding::Q4ChunkMajor: return qweight(q4_chunk_major(name, shape));
        case StrixwEncoding::Q8ChunkMajor: return qweight(q8_chunk_major(name, shape));
        default:
            STRIX_FAIL("StrixwDevice: '", name, "' in '", file_->path(), "' is ", strixw_encoding_name(t.encoding), " ",
                       shape_str(t.shape), ", expected q4 or q8 chunk-major ", shape_str(shape));
    }
}

const float *StrixwDevice::f32(const std::string &name, const std::vector<int64_t> &shape) const {
    return static_cast<const float *>(ptr(tensor(name, StrixwEncoding::F32, shape), StrixwRole::Data));
}

const uint16_t *StrixwDevice::bf16(const std::string &name, const std::vector<int64_t> &shape) const {
    return static_cast<const uint16_t *>(ptr(tensor(name, StrixwEncoding::BF16, shape), StrixwRole::Data));
}

}  // namespace strix
