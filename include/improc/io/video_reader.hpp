// include/improc/io/video_reader.hpp
#pragma once

#include <filesystem>
#include <optional>
#include <opencv2/videoio.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::io {

using improc::core::Image;
using improc::core::BGR;

/**
 * @brief Synchronous sequential video file reader.
 *
 * Opens the file at construction; throws if the file does not exist or
 * cannot be decoded as video. Call `next()` to read frames one-by-one.
 * Non-copyable and non-movable. RAII: destructor closes the file.
 *
 * @throws improc::FileNotFoundError if the path does not exist.
 * @throws improc::IoError           if the file cannot be opened as video.
 *
 * @code
 * VideoReader reader{"clip.avi"};
 * while (auto frame = reader.next()) {
 *     Image<BGR> img = *frame;
 * }
 * @endcode
 */
class VideoReader {
public:
    /// @brief Opens the video file. Throws on failure.
    explicit VideoReader(std::filesystem::path path);
    /// @brief Closes the file (RAII).
    ~VideoReader() = default;

    /// @brief Deleted copy constructor — non-copyable.
    VideoReader(const VideoReader&)            = delete;
    /// @brief Deleted copy assignment — non-copyable.
    VideoReader& operator=(const VideoReader&) = delete;
    /// @brief Deleted move constructor — non-movable.
    VideoReader(VideoReader&&)                 = delete;
    /// @brief Deleted move assignment — non-movable.
    VideoReader& operator=(VideoReader&&)      = delete;

    /// @brief Reads and returns the next frame, or `std::nullopt` at EOF or on read error.
    std::optional<Image<BGR>> next();

    /// @brief Closes the underlying capture; subsequent `next()` calls return `std::nullopt`.
    void close();

    /// @brief Returns `true` if the file is open and ready to read.
    bool is_open() const { return cap_.isOpened(); }

    /// @brief Playback frame rate reported by the container. May be approximate.
    double fps() const { return cap_.get(cv::CAP_PROP_FPS); }

    /**
     * @brief Total frame count reported by the container.
     * @note Returns -1 for containers where this value is unreliable (e.g. some .mkv files).
     */
    int frame_count() const {
        return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    }

    /// @brief Frame width in pixels.
    int width()  const { return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));  }
    /// @brief Frame height in pixels.
    int height() const { return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)); }

private:
    cv::VideoCapture cap_;
};

} // namespace improc::io
