//
// Created by Michał Maj on 09/04/2025.
//

#include "improc/io/camera_capture.hpp"
#include <iostream>

namespace improc::io {

CameraCapture::CameraCapture(unsigned int camera_id)
    : camera_id_(camera_id), keep_running_(true), stopped_(false) {
  start();
}

CameraCapture::~CameraCapture() {
  stop();
}

void CameraCapture::stop() {
  if (stopped_) return;

  stopped_ = true;
  keep_running_ = false;

  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }

  if (cap_.isOpened()) {
    cap_.release();
  }
}

cv::Mat CameraCapture::getFrame() {
  std::shared_lock<std::shared_mutex> lock(frame_mutex_);
  return frame_.clone();
}

const std::string& CameraCapture::getWindowName() const {
  return window_name_;
}

void CameraCapture::start() {
  capture_thread_ = std::thread(&CameraCapture::captureLoop, this);
}

void CameraCapture::captureLoop() {
  cap_.open(camera_id_);
  if (!cap_.isOpened()) {
    std::cerr << "Cannot open camera " << camera_id_ << std::endl;
    return;
  }

  while (keep_running_) {
    cv::Mat temp_frame;
    cap_.read(temp_frame);

    if (temp_frame.empty()) continue;

    std::unique_lock<std::shared_mutex> lock(frame_mutex_);
    frame_ = std::move(temp_frame);
  }
}

} // namespace improc::io
