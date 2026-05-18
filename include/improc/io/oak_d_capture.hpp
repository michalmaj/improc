// include/improc/io/oak_d_capture.hpp
#pragma once
#include "improc/io/camera_frame.hpp"
#include "improc/io/camera_source.hpp"
#include "improc/error.hpp"
#include <atomic>
#include <expected>
#include <memory>
#include <stdexcept>
#include <string>

#ifdef IMPROC_WITH_DEPTHAI
#include <depthai/depthai.hpp>
#endif

namespace improc::io {

// OakDCapture — streams aligned RGB + metric depth from an OAK-D camera.
// Only available when compiled with -DIMPROC_WITH_DEPTHAI=ON.
// Satisfies CameraSourceType<OakDCapture>.
//
// Usage:
//   OakDCapture oak;                          // first found device
//   OakDCapture oak("14442C10D13B8D1300");   // specific MX ID
//   oak.start();
//   auto frame = oak.getFrame();
//   oak.stop();
//
// getFrame() blocks until both RGB and depth are available (up to timeout_ms_).
// Returns Error::Code::Timeout if either queue times out.
// Returns Error::Code::CameraUnavailable if start() was not called.

#ifdef IMPROC_WITH_DEPTHAI

class OakDCapture {
public:
    // Constructs OakDCapture targeting the first available device.
    OakDCapture();

    // Constructs OakDCapture targeting a specific device by MX ID.
    explicit OakDCapture(std::string mx_id);

    ~OakDCapture();

    OakDCapture(const OakDCapture&) = delete;
    OakDCapture& operator=(const OakDCapture&) = delete;
    OakDCapture(OakDCapture&&) = delete;
    OakDCapture& operator=(OakDCapture&&) = delete;

    // Opens the device and starts the pipeline. Idempotent.
    void start();

    // Stops the pipeline and closes the device. Idempotent.
    void stop();

    // Blocks until an aligned RGB+depth pair is available, up to timeout_ms_.
    // Returns a CameraFrame with both rgb and depth populated on success.
    // Returns Error::Code::Timeout if either queue times out.
    // Returns Error::Code::CameraUnavailable if not started.
    //
    // Thread safety: getFrame() and stop() must not be called concurrently.
    // FramePipeline guarantees this — it stops the pipeline before joining workers.
    std::expected<CameraFrame, improc::Error> getFrame();

private:
    std::string mx_id_;          // empty = use first device
    std::string source_id_;      // pre-computed "oak-d" or "oak-d:<mxid>"
    int timeout_ms_{1000};       // per-queue timeout in ms
    std::atomic<bool> started_{false};

    std::shared_ptr<dai::Device>          device_;
    std::shared_ptr<dai::DataOutputQueue> rgb_queue_;
    std::shared_ptr<dai::DataOutputQueue> depth_queue_;
};

static_assert(CameraSourceType<OakDCapture>);

#else  // !IMPROC_WITH_DEPTHAI

// Stub that fails at runtime when OAK-D support was not compiled in.
// Provides the same interface so code can be written against it and
// the build failure only happens at the call site when depthai is OFF.
class OakDCapture {
public:
    OakDCapture() = default;
    explicit OakDCapture(std::string) {}
    void start()  { throw std::runtime_error("OakDCapture: built without -DIMPROC_WITH_DEPTHAI=ON"); }
    void stop()   {}
    std::expected<CameraFrame, improc::Error> getFrame() {
        return std::unexpected(improc::Error::camera_unavailable(
            "OakDCapture: built without -DIMPROC_WITH_DEPTHAI=ON"));
    }
};

static_assert(CameraSourceType<OakDCapture>);

#endif  // IMPROC_WITH_DEPTHAI

}  // namespace improc::io
