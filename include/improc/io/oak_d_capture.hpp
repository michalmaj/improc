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

/**
 * @brief Streams aligned RGB + metric depth from a Luxonis OAK-D camera.
 *
 * Requires the library to be built with `-DIMPROC_WITH_DEPTHAI=ON`.
 * Without it, all methods throw `std::runtime_error` at runtime.
 * Satisfies `CameraSourceType<OakDCapture>`.
 *
 * @code
 * OakDCapture oak;   // first available device
 * oak.start();
 * if (auto f = oak.getFrame()) {
 *     cv::imshow("rgb",   f->rgb.mat());
 *     cv::imshow("depth", f->depth->mat());
 * }
 * oak.stop();
 * @endcode
 */

#ifdef IMPROC_WITH_DEPTHAI

class OakDCapture {
public:
    /// @brief Constructs targeting the first available OAK-D device.
    OakDCapture();

    /// @brief Constructs targeting a specific device identified by MX ID.
    explicit OakDCapture(std::string mx_id);

    ~OakDCapture();

    OakDCapture(const OakDCapture&) = delete;
    OakDCapture& operator=(const OakDCapture&) = delete;
    OakDCapture(OakDCapture&&) = delete;
    OakDCapture& operator=(OakDCapture&&) = delete;

    /// @brief Opens the device and starts the DepthAI pipeline. Idempotent.
    void start();

    /// @brief Stops the pipeline and closes the device. Idempotent.
    void stop();

    /// @brief Returns an aligned RGB+depth frame, blocking up to 1000 ms per queue.
    /// @return Error::Code::Timeout if either queue times out; CameraUnavailable if not started.
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

/// @brief Stub — OAK-D support not compiled in. All methods throw `std::runtime_error`.
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
