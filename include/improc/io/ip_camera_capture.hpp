// include/improc/io/ip_camera_capture.hpp
#pragma once

#include <atomic>
#include <expected>
#include <shared_mutex>
#include <string>
#include <thread>
#include <opencv2/videoio.hpp>
#include "improc/io/camera_frame.hpp"
#include "improc/error.hpp"

namespace improc::io {

/**
 * @brief Threaded RTSP / HTTP camera source that streams from a URL.
 *
 * Starts a background thread on `start()` that continuously reads frames
 * via `cv::VideoCapture`. `getFrame()` returns the most recent decoded frame.
 * Non-copyable and non-movable (owns a `std::thread`).
 *
 * @code
 * IPCameraCapture cam("rtsp://192.168.1.100:554/stream");
 * cam.start();
 * auto frame = cam.getFrame();
 * cam.stop();
 * @endcode
 */
class IPCameraCapture {
public:
    /// @brief Constructs the capture for the given URL (RTSP, HTTP, or any OpenCV-compatible URI).
    explicit IPCameraCapture(std::string url);
    ~IPCameraCapture();

    IPCameraCapture(const IPCameraCapture&) = delete;
    IPCameraCapture& operator=(const IPCameraCapture&) = delete;
    IPCameraCapture(IPCameraCapture&&) = delete;
    IPCameraCapture& operator=(IPCameraCapture&&) = delete;

    /// @brief Opens the stream and starts the background capture thread.
    void start();
    /// @brief Stops the capture thread and closes the stream.
    void stop();
    /// @brief Returns the most recently decoded frame.
    /// @return CameraUnavailable if start() was not called or the stream has no frames yet.
    std::expected<CameraFrame, improc::Error> getFrame();

private:
    void captureLoop();

    std::string url_;
    std::string source_id_;

    std::atomic<bool> started_{false};
    std::atomic<bool> keep_running_{false};
    std::atomic<bool> camera_available_{false};

    std::thread capture_thread_;
    cv::VideoCapture cap_;
    cv::Mat last_frame_;
    std::shared_mutex frame_mutex_;
};

} // namespace improc::io
