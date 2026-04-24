// src/io/video_reader.cpp
#include "improc/io/video_reader.hpp"
#include <filesystem>

namespace improc::io {

VideoReader::VideoReader(std::filesystem::path path) {
    if (!std::filesystem::exists(path))
        throw FileNotFoundError{path};

    if (!cap_.open(path.string()))
        throw IoError{"Cannot open '" + path.string() + "' as a video file"};
}

std::optional<Image<BGR>> VideoReader::next() {
    if (!cap_.isOpened()) return std::nullopt;
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) return std::nullopt;
    return Image<BGR>(frame);
}

void VideoReader::close() {
    cap_.release();
}

} // namespace improc::io
