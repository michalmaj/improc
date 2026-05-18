// src/io/webcam_capture.cpp
#include "improc/io/webcam_capture.hpp"

namespace improc::io {

WebcamCapture::WebcamCapture(int device_id)
    : device_(device_id)
    , source_id_("webcam:" + std::to_string(device_id)) {}

WebcamCapture::WebcamCapture(std::string device_path)
    : device_(device_path)
    , source_id_("webcam:" + device_path) {}

WebcamCapture::~WebcamCapture() { stop(); }

void WebcamCapture::start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;  // idempotent
    keep_running_ = true;
    capture_thread_ = std::thread(&WebcamCapture::captureLoop, this);
}

void WebcamCapture::stop() {
    if (!started_) return;
    keep_running_ = false;
    if (capture_thread_.joinable()) capture_thread_.join();
    if (cap_.isOpened()) cap_.release();
}

std::expected<CameraFrame, improc::Error> WebcamCapture::getFrame() {
    if (!camera_available_) {
        int id = std::holds_alternative<int>(device_) ? std::get<int>(device_) : -1;
        return std::unexpected(improc::Error::camera_unavailable(id));
    }
    std::shared_lock lock(frame_mutex_);
    if (last_frame_.empty()) {
        int id = std::holds_alternative<int>(device_) ? std::get<int>(device_) : -1;
        return std::unexpected(improc::Error::camera_frame_empty(id));
    }
    CameraFrame frame;
    frame.rgb = core::Image<core::BGR>(last_frame_.clone());
    frame.timestamp = std::chrono::steady_clock::now();
    frame.source_id = source_id_;
    return frame;
}

const std::string& WebcamCapture::getWindowName() const { return window_name_; }

void WebcamCapture::captureLoop() {
    std::visit([this](auto& dev) { cap_.open(dev); }, device_);
    if (!cap_.isOpened()) return;
    camera_available_ = true;
    while (keep_running_) {
        cv::Mat tmp;
        cap_.read(tmp);
        if (tmp.empty()) continue;
        std::unique_lock lock(frame_mutex_);
        last_frame_ = std::move(tmp);
    }
}

} // namespace improc::io
