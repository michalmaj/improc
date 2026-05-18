// src/io/ip_camera_capture.cpp
#include "improc/io/ip_camera_capture.hpp"

namespace improc::io {

IPCameraCapture::IPCameraCapture(std::string url)
    : url_(std::move(url)), source_id_(url_) {}

IPCameraCapture::~IPCameraCapture() { stop(); }

void IPCameraCapture::start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;
    keep_running_ = true;
    capture_thread_ = std::thread(&IPCameraCapture::captureLoop, this);
}

void IPCameraCapture::stop() {
    if (!started_) return;
    keep_running_ = false;
    if (capture_thread_.joinable()) capture_thread_.join();
    if (cap_.isOpened()) cap_.release();
    camera_available_ = false;
    started_ = false;
}

std::expected<CameraFrame, improc::Error> IPCameraCapture::getFrame() {
    if (!camera_available_)
        return std::unexpected(improc::Error{
            improc::Error::Code::CameraUnavailable,
            "IP camera '" + url_ + "' is not available"});
    std::shared_lock lock(frame_mutex_);
    if (last_frame_.empty())
        return std::unexpected(improc::Error{
            improc::Error::Code::CameraFrameEmpty,
            "IP camera '" + url_ + "' returned an empty frame"});
    CameraFrame frame;
    frame.rgb = core::Image<core::BGR>(last_frame_.clone());
    frame.timestamp = std::chrono::steady_clock::now();
    frame.source_id = source_id_;
    return frame;
}

void IPCameraCapture::captureLoop() {
    cap_.open(url_);
    if (!cap_.isOpened()) return;
    camera_available_ = true;
    while (keep_running_) {
        cv::Mat tmp;
        cap_.read(tmp);
        if (tmp.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        std::unique_lock lock(frame_mutex_);
        last_frame_ = std::move(tmp);
    }
}

} // namespace improc::io
