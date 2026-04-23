//
// Created by Michał Maj on 09/04/2025.
//

#pragma once

#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <expected>
#include <shared_mutex>
#include <string>
#include "improc/error.hpp"

namespace improc::io {

/**
 * @brief Asynchronous threaded camera frame capture.
 *
 * Starts a background thread on construction that continuously reads frames
 * from the specified camera index. `getFrame()` returns the latest captured
 * frame or an error if the camera is unavailable. Non-copyable and non-movable.
 *
 * @code
 * CameraCapture cam(0);
 * while (true) {
 *     auto frame = cam.getFrame();
 *     if (!frame) continue;
 *     // process *frame ...
 * }
 * @endcode
 */
class CameraCapture {
public:
  /// @brief Constructs and starts the capture thread for the given camera index.
  explicit CameraCapture(unsigned int camera_id);
  /// @brief Stops the capture thread and releases the camera device.
  ~CameraCapture();

  /// @brief Deleted copy constructor — non-copyable.
  CameraCapture(const CameraCapture&) = delete;
  /// @brief Deleted move constructor — non-movable.
  CameraCapture(CameraCapture&&) = delete;
  /// @brief Deleted copy assignment — non-copyable.
  CameraCapture& operator=(const CameraCapture&) = delete;
  /// @brief Deleted move assignment — non-movable.
  CameraCapture& operator=(CameraCapture&&) = delete;

  /// @brief Signals the capture thread to stop and blocks until it joins.
  void stop();

  /// @brief Returns the latest captured frame, or an error if the camera is
  ///        unavailable or has not produced a frame yet.
  std::expected<cv::Mat, improc::Error> getFrame();

  /// @brief Returns the display window name associated with this capture.
  const std::string& getWindowName() const;

private:
  void start();
  void captureLoop();

  unsigned int      camera_id_;
  std::atomic<bool> keep_running_;
  std::atomic<bool> stopped_;
  std::atomic<bool> camera_available_{false};
  std::thread       capture_thread_;
  cv::VideoCapture  cap_;
  cv::Mat           frame_;
  std::shared_mutex frame_mutex_;
  std::string       window_name_ = "Frame";
};

} // namespace improc::io

