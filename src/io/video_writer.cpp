// src/io/video_writer.cpp
#include "improc/io/video_writer.hpp"
#include <format>

namespace improc::io {

VideoWriter::VideoWriter(std::filesystem::path path)
    : path_(std::move(path))
{}

VideoWriter::~VideoWriter() {
    close();
}

void VideoWriter::close() {
    if (writer_.isOpened())
        writer_.release();
}

std::string VideoWriter::codec_from_path(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (ext == ".mp4" || ext == ".m4v" || ext == ".mov") return "mp4v";
    if (ext == ".avi")                                    return "MJPG";
    if (ext == ".mkv")                                    return "XVID";
    return "mp4v";  // reasonable fallback
}

void VideoWriter::open(const cv::Size& frame_size) {
    const std::string fourcc_str = codec_.empty() ? codec_from_path(path_) : codec_;
    const int fourcc = cv::VideoWriter::fourcc(
        fourcc_str[0], fourcc_str[1], fourcc_str[2], fourcc_str[3]);

    writer_.open(path_.string(), fourcc, fps_, frame_size);
    if (!writer_.isOpened())
        throw IoError{std::format(
            "VideoWriter: failed to open '{}' (codec={}, fps={}, size={}x{})",
            path_.string(), fourcc_str, fps_, frame_size.width, frame_size.height)};
}

Image<BGR> VideoWriter::operator()(Image<BGR> img) {
    const cv::Size frame_size{
        width_  > 0 ? width_  : img.cols(),
        height_ > 0 ? height_ : img.rows()
    };

    if (!writer_.isOpened())
        open(frame_size);

    if (img.cols() != frame_size.width || img.rows() != frame_size.height) {
        throw IoError{std::format(
            "VideoWriter: frame size {}x{} does not match writer size {}x{}",
            img.cols(), img.rows(), frame_size.width, frame_size.height)};
    }

    writer_.write(img.mat());
    return img;
}

} // namespace improc::io
