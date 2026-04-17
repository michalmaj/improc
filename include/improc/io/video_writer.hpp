// include/improc/io/video_writer.hpp
#pragma once

#include <filesystem>
#include <string>
#include <opencv2/videoio.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::io {

using improc::core::Image;
using improc::core::BGR;

// Synchronous video writer with RAII semantics.
// Size is auto-detected from the first frame if not set explicitly.
// Codec is auto-detected from file extension if not set explicitly.
//   .mp4 / .mov  → mp4v
//   .avi         → MJPG
//   .mkv         → XVID
//
// Pipeline-compatible: operator() writes the frame and returns it unchanged.
//
//   VideoWriter writer{"out.mp4"};
//   writer.fps(30);
//   img | Show{"preview"} | writer;   // pipeline form
//   writer(img);                       // direct form
struct VideoWriter {
    explicit VideoWriter(std::filesystem::path path);
    ~VideoWriter();

    VideoWriter(const VideoWriter&)            = delete;
    VideoWriter& operator=(const VideoWriter&) = delete;
    VideoWriter(VideoWriter&&)                 = delete;
    VideoWriter& operator=(VideoWriter&&)      = delete;

    VideoWriter& fps(double f) {
        if (f <= 0.0) throw ParameterError{"fps", "must be positive", "VideoWriter"};
        fps_ = f;
        return *this;
    }

    // Explicit output size. If not set, detected from the first frame.
    VideoWriter& size(int width, int height) {
        if (width <= 0)  throw ParameterError{"width",  "must be positive", "VideoWriter"};
        if (height <= 0) throw ParameterError{"height", "must be positive", "VideoWriter"};
        width_  = width;
        height_ = height;
        return *this;
    }

    // Override the auto-detected codec. Pass a 4-character FourCC string, e.g. "mp4v".
    VideoWriter& codec(std::string fourcc) {
        if (fourcc.size() != 4)
            throw ParameterError{"codec", "must be a 4-character FourCC string", "VideoWriter"};
        codec_ = std::move(fourcc);
        return *this;
    }

    // Write one frame. Returns the frame unchanged so the call composes in pipelines.
    // On the first call, opens the underlying cv::VideoWriter (auto-sizing if needed).
    // Throws IoError if the writer cannot be opened or the frame has a different size
    // than the writer was initialized with.
    Image<BGR> operator()(Image<BGR> img);

    // Release the writer and finalise the file. Safe to call multiple times.
    void close();

    bool is_open() const { return writer_.isOpened(); }

private:
    void open(const cv::Size& frame_size);
    static std::string codec_from_path(const std::filesystem::path& path);

    std::filesystem::path path_;
    std::string           codec_;    // empty → auto from extension
    double                fps_    = 25.0;
    int                   width_  = 0;   // 0 → detect from first frame
    int                   height_ = 0;
    cv::VideoWriter       writer_;
};

} // namespace improc::io
