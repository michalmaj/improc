//
// Created by Michał Maj on 09/04/2025.
//

#pragma once

#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <shared_mutex>
#include <string>

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
  cv::Mat getFrame();
  const std::string& getWindowName() const;

private:
  void start();
  void captureLoop();

  unsigned int camera_id_;
  std::atomic<bool> keep_running_;
  std::atomic<bool> stopped_;
  std::thread capture_thread_;
  cv::VideoCapture cap_;
  cv::Mat frame_;
  std::shared_mutex frame_mutex_;
  std::string window_name_ = "Frame";
};

} // namespace improc::io

