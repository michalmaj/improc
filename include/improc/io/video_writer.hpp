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

/**
 * @brief RAII video file writer with lazy open on first frame.
 *
 * Codec is auto-detected from the file extension:
 * `.mp4`/`.mov` → `mp4v`, `.avi` → `MJPG`, `.mkv` → `XVID`.
 * The writer opens on the first frame to infer frame dimensions.
 * Supports pipeline use via `operator()(Image<BGR>)`.
 *
 * @code
 * VideoWriter writer{"output.mp4"};
 * writer.fps(30);
 * frame | writer;
 * @endcode
 */
struct VideoWriter {
    /// @brief Constructs the writer for the given output file path; does not open the file yet.
    explicit VideoWriter(std::filesystem::path path);
    /// @brief Finalises and closes the video file.
    ~VideoWriter();

    /// @brief Deleted copy constructor — non-copyable.
    VideoWriter(const VideoWriter&)            = delete;
    /// @brief Deleted copy assignment — non-copyable.
    VideoWriter& operator=(const VideoWriter&) = delete;
    /// @brief Deleted move constructor — non-movable.
    VideoWriter(VideoWriter&&)                 = delete;
    /// @brief Deleted move assignment — non-movable.
    VideoWriter& operator=(VideoWriter&&)      = delete;

    /// @brief Sets the output frame rate.
    /// @throws improc::ParameterError if `f` <= 0.
    VideoWriter& fps(double f) {
        if (f <= 0.0) throw ParameterError{"fps", "must be positive", "VideoWriter"};
        fps_ = f;
        return *this;
    }

    /// @brief Sets an explicit output frame size; if not called, size is inferred from the first frame.
    /// @throws improc::ParameterError if width or height <= 0.
    VideoWriter& size(int width, int height) {
        if (width <= 0)  throw ParameterError{"width",  "must be positive", "VideoWriter"};
        if (height <= 0) throw ParameterError{"height", "must be positive", "VideoWriter"};
        width_  = width;
        height_ = height;
        return *this;
    }

    /// @brief Overrides the auto-detected codec; pass a 4-character FourCC string (e.g. `"mp4v"`).
    /// @throws improc::ParameterError if `fourcc` is not exactly 4 characters.
    VideoWriter& codec(std::string fourcc) {
        if (fourcc.size() != 4)
            throw ParameterError{"codec", "must be a 4-character FourCC string", "VideoWriter"};
        codec_ = std::move(fourcc);
        return *this;
    }

    /// @brief Writes one frame and returns it unchanged, enabling pipeline composition.
    ///
    /// On the first call the underlying `cv::VideoWriter` is opened (auto-sizing if needed).
    /// @throws improc::IoError if the writer cannot be opened or the frame size mismatches.
    Image<BGR> operator()(Image<BGR> img);

    /// @brief Releases the writer and finalises the file. Safe to call multiple times.
    void close();

    /// @brief Returns `true` if the underlying `cv::VideoWriter` is open.
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
