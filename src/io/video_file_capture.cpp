// src/io/video_file_capture.cpp
#include "improc/io/video_file_capture.hpp"
#include "improc/io/video_reader.hpp"

namespace improc::io {

VideoFileCapture::VideoFileCapture(std::filesystem::path path)
    : path_(std::move(path))
    , source_id_(path_.string()) {}

VideoFileCapture::~VideoFileCapture() { stop(); }

void VideoFileCapture::start() {
    bool expected = false;
    if (!started_.compare_exchange_strong(expected, true)) return;
    try {
        reader_ = std::make_unique<VideoReader>(path_);
    } catch (...) {
        started_ = false;
        throw;
    }
}

void VideoFileCapture::stop() {
    if (!started_.exchange(false)) return;
    reader_.reset();
}

std::expected<CameraFrame, improc::Error> VideoFileCapture::getFrame() {
    if (!started_ || !reader_) {
        return std::unexpected(improc::Error::camera_unavailable(source_id_));
    }
    auto maybe = reader_->next();
    if (!maybe) {
        return std::unexpected(improc::Error::end_of_file(source_id_));
    }
    CameraFrame frame;
    frame.rgb       = std::move(*maybe);
    frame.timestamp = std::chrono::steady_clock::now();
    frame.source_id = source_id_;
    return frame;
}

}  // namespace improc::io
