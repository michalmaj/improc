// include/improc/io/video_file_capture.hpp
#pragma once

#include <atomic>
#include <expected>
#include <filesystem>
#include <memory>
#include "improc/io/camera_frame.hpp"
#include "improc/io/camera_source.hpp"
#include "improc/error.hpp"

namespace improc::io {

class VideoReader;  // forward-declared; full type included in .cpp

/**
 * @brief Reads frames from a video file as a `CameraSourceType`.
 *
 * Wraps `VideoReader` so that video files work in `FramePipeline` identically
 * to live cameras. Non-copyable and non-movable.
 *
 * @throws improc::FileNotFoundError from `start()` if the file does not exist.
 * @throws improc::IoError           from `start()` if the file cannot be decoded as video.
 *
 * @code
 * VideoFileCapture cap("clip.mp4");
 * cap.start();
 * while (auto f = cap.getFrame()) {
 *     process(*f->rgb);
 * }
 * cap.stop();
 * @endcode
 */
class VideoFileCapture {
public:
    explicit VideoFileCapture(std::filesystem::path path);
    ~VideoFileCapture();

    VideoFileCapture(const VideoFileCapture&)            = delete;
    VideoFileCapture& operator=(const VideoFileCapture&) = delete;
    VideoFileCapture(VideoFileCapture&&)                 = delete;
    VideoFileCapture& operator=(VideoFileCapture&&)      = delete;

    /**
     * @brief Opens the video file. Idempotent.
     * @throws improc::FileNotFoundError if the path does not exist.
     * @throws improc::IoError           if the file cannot be opened as video.
     */
    void start();

    /// @brief Closes the video file. Idempotent. Also called by destructor.
    void stop();

    /**
     * @brief Returns the next frame from the video file.
     *
     * Returns `Error::CameraUnavailable` if `start()` has not been called.
     * Returns `Error::EndOfFile` when all frames have been read.
     */
    std::expected<CameraFrame, improc::Error> getFrame();

private:
    std::filesystem::path        path_;
    std::string                  source_id_;
    std::atomic<bool>            started_{false};
    std::unique_ptr<VideoReader> reader_;
};

static_assert(CameraSourceType<VideoFileCapture>);

}  // namespace improc::io
