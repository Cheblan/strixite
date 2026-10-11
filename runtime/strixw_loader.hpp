#pragma once

// strixw weights resident on the device (formats/strixw.hpp): the file's whole data region is one device
// allocation, filled by concurrent pread() into two pinned staging buffers and async copies, each staging
// buffer reused only after an event says its previous copy finished (the gfx1151 host-buffer hazard:
// reusing a pinned buffer for the next async copy without a gate corrupts data). Tensors keep their file
// offsets, so a component's device pointer is base + (offset - data offset) - alignment carries over (4096
// per tensor, 256 per component). On gfx1151 hipMalloc comes out of GTT (system RAM: 124 GiB of a 128 GB machine).
//
// WeightPlacement::HostExperts (test / bench tools only, never the server): the
// routed + shared expert stacks (`mlp.experts_gate_up`, `mlp.experts_down`, ~60 GB of the served file) go to one
// pinned, mapped, non-coherent host allocation the kernels read over PCIe, everything else to the device buffer -
// so a discrete card smaller than the file runs the same kernels on the same bytes. The file is split into runs of
// consecutive tensors with one placement; each run keeps its file offset mod 4096, so alignment still carries over.
// The host allocation is refused unless MemAvailable keeps host_margin_gib after it (pinned memory can't be
// swapped; a machine at ~0 MemAvailable hard-hangs). On unified memory it makes no difference beyond the copy.

#include "formats/strixw.hpp"
#include "runtime/device_buffer.hpp"
#include "runtime/q4_device.hpp"
#include "runtime/q5_device.hpp"
#include "runtime/q6_device.hpp"
#include "runtime/q8_device.hpp"
#include "runtime/qweight.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strix {

enum class WeightPlacement { Device, HostExperts };
const char *weight_placement_name(WeightPlacement p);
// "device" / "host-experts" (a tool option's value; `what` names it in the error).
WeightPlacement parse_weight_placement(const std::string &v, const char *what);
// True for the tensors HostExperts places in host memory.
bool is_host_expert_tensor(const std::string &name);

class StrixwDevice {
public:
    // Loads path. Throws on any header/index problem (StrixwFile validates) or I/O / HIP failure. Component
    // hashes aren't re-checked (inspect_strixw --verify does that); the header and index hashes are.
    explicit StrixwDevice(const std::string &path, int read_threads = 8,
                          WeightPlacement placement = WeightPlacement::Device, double host_margin_gib = 24);
    StrixwDevice(const StrixwDevice &) = delete;
    StrixwDevice &operator=(const StrixwDevice &) = delete;

    const StrixwFile &file() const { return *file_; }
    double load_seconds() const { return load_seconds_; }
    uint64_t data_bytes() const { return device_bytes_ + host_bytes_; }
    WeightPlacement placement() const { return placement_; }
    uint64_t device_bytes() const { return device_bytes_; }  // the device buffer (region sizes, padding included)
    uint64_t host_bytes() const { return host_bytes_; }      // the pinned host allocation (0 for Device)

    // Device pointers, checked: the tensor must exist with the expected encoding and shape.
    const StrixwTensor &tensor(const std::string &name, StrixwEncoding enc, const std::vector<int64_t> &shape) const;
    const void *ptr(const StrixwTensor &t, StrixwRole role) const;
    Q4DeviceView q4(const std::string &name, const std::vector<int64_t> &shape) const;
    Q8DeviceView q8(const std::string &name, const std::vector<int64_t> &shape) const;
    Q6DeviceView q6(const std::string &name, const std::vector<int64_t> &shape) const;
    Q5DeviceView q5(const std::string &name, const std::vector<int64_t> &shape) const;
    // A dense projection stored as Q4, Q5, Q6 or Q8 row-major, whichever the file has (anything else throws).
    QWeightView qw(const std::string &name, const std::vector<int64_t> &shape) const;
    Q4ChunkMajorView q4_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const;
    Q8ChunkMajorView q8_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const;
    // A chunk-major weight stored as Q4 or Q8, whichever the file has (anything else throws).
    QChunkMajorView qw_chunk_major(const std::string &name, const std::vector<int64_t> &shape) const;
    const float *f32(const std::string &name, const std::vector<int64_t> &shape) const;
    const uint16_t *bf16(const std::string &name, const std::vector<int64_t> &shape) const;

private:
    // A byte range [start, end) of the file (absolute offsets) at offset `at` of the device or host region.
    struct Run {
        uint64_t start = 0, end = 0, at = 0;
        bool host = false;
    };
    std::unique_ptr<StrixwFile> file_;
    WeightPlacement placement_ = WeightPlacement::Device;
    std::vector<Run> runs_;  // by start, covering the data region
    DeviceBuffer<uint8_t> data_;
    struct HostFree {
        void operator()(void *p) const;
    };
    std::unique_ptr<void, HostFree> host_;  // hipHostMalloc'd (HostExperts only)
    const uint8_t *host_dev_ = nullptr;  // its device pointer
    uint64_t device_bytes_ = 0, host_bytes_ = 0;
    double load_seconds_ = 0;
};

}  // namespace strix
