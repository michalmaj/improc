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

class CameraCapture {
public:
  explicit CameraCapture(unsigned int camera_id);
  ~CameraCapture();

  CameraCapture(const CameraCapture&) = delete;
  CameraCapture(CameraCapture&&) = delete;
  CameraCapture& operator=(const CameraCapture&) = delete;
  CameraCapture& operator=(CameraCapture&&) = delete;

  void stop();

  // Returns the latest captured frame, or an Error if the camera is
  // unavailable or has not produced a frame yet.
  std::expected<cv::Mat, improc::Error> getFrame();

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

