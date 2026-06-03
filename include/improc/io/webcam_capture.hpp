// include/improc/io/webcam_capture.hpp
#pragma once

#include <atomic>
#include <expected>
#include <shared_mutex>
#include <string>
#include <thread>
#include <variant>
#include <opencv2/videoio.hpp>
#include "improc/io/camera_frame.hpp"
#include "improc/error.hpp"

namespace improc::io {

/**
 * @brief Asynchronous webcam capture with explicit start/stop lifecycle.
 *
 * The background thread is NOT started on construction.
 * Call `start()` explicitly after construction. `start()` is idempotent: calling
 * it on an already-started capture is a no-op. Non-copyable and non-movable.
 *
 * Supports both integer device indices (e.g. `0` for `/dev/video0`) and
 * string device paths (e.g. `"/dev/video1"`).
 *
 * @code
 * WebcamCapture camera(0);
 * camera.start();
 * // ... later ...
 * auto frame = camera.getFrame();
 * if (frame) {
 *     cv::imshow("preview", frame->rgb->mat());
 * }
 * camera.stop();
 * @endcode
 */
class WebcamCapture {
public:
    /// @brief Constructs the capture for the given integer device index (e.g. 0).
    explicit WebcamCapture(int device_id);

    /// @brief Constructs the capture for the given device path (e.g. "/dev/video1").
    explicit WebcamCapture(std::string device_path);

    /// @brief Stops the capture thread (if running) and releases the camera device.
    ~WebcamCapture();

    /// @brief Deleted copy constructor — non-copyable.
    WebcamCapture(const WebcamCapture&) = delete;
    /// @brief Deleted copy assignment — non-copyable.
    WebcamCapture& operator=(const WebcamCapture&) = delete;
    /// @brief Deleted move constructor — non-movable.
    WebcamCapture(WebcamCapture&&) = delete;
    /// @brief Deleted move assignment — non-movable.
    WebcamCapture& operator=(WebcamCapture&&) = delete;

    /**
     * @brief Starts the background capture thread.
     *
     * Idempotent: calling `start()` on an already-started capture is a no-op.
     * Thread-safe relative to the caller; not safe to call concurrently from
     * multiple threads.
     */
    void start();

    /**
     * @brief Signals the background thread to stop and blocks until it joins.
     *
     * Idempotent: safe to call multiple times or when never started.
     * Also called automatically by the destructor.
     */
    void stop();

    /**
     * @brief Returns the latest captured frame.
     *
     * Returns `std::unexpected(Error::camera_unavailable(...))` if the device
     * could not be opened, or `std::unexpected(Error::camera_frame_empty(...))`
     * if no frame has been captured yet.
     */
    std::expected<CameraFrame, improc::Error> getFrame();

    /// @brief Returns the display window name associated with this capture.
    const std::string& getWindowName() const;

private:
    void captureLoop();

    std::variant<int, std::string> device_;
    std::string                    source_id_;
    std::string                    window_name_ = "Frame";

    std::atomic<bool>  started_{false};
    std::atomic<bool>  keep_running_{false};
    std::atomic<bool>  camera_available_{false};

    std::thread        capture_thread_;
    cv::VideoCapture   cap_;
    cv::Mat            last_frame_;
    std::shared_mutex  frame_mutex_;
};

} // namespace improc::io
