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

class IPCameraCapture {
public:
    explicit IPCameraCapture(std::string url);
    ~IPCameraCapture();

    IPCameraCapture(const IPCameraCapture&) = delete;
    IPCameraCapture& operator=(const IPCameraCapture&) = delete;
    IPCameraCapture(IPCameraCapture&&) = delete;
    IPCameraCapture& operator=(IPCameraCapture&&) = delete;

    void start();
    void stop();
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
