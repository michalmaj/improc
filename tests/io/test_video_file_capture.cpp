// tests/io/test_video_file_capture.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <cstdlib>
#include <string>
#include "improc/io/video_file_capture.hpp"
#include "improc/io/any_camera_source.hpp"
#include "improc/io/camera_source.hpp"
#include "improc/exceptions.hpp"

using improc::io::VideoFileCapture;
using improc::io::CameraSourceType;
using improc::Error;
namespace fs = std::filesystem;

namespace {

struct TempFile {
    explicit TempFile(fs::path p) : path(std::move(p)) {}
    ~TempFile() { fs::remove(path); }
    fs::path path;
};

bool make_test_video(const fs::path& path, int frames = 8,
                     int w = 64, int h = 48) {
    const char* ffmpeg_paths[] = {
        "ffmpeg", "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg", nullptr
    };
    std::string ffmpeg_bin;
    for (const char** p = ffmpeg_paths; *p; ++p) {
        if (std::system((std::string(*p) + " -version >/dev/null 2>&1").c_str()) == 0) {
            ffmpeg_bin = *p; break;
        }
    }
    if (ffmpeg_bin.empty()) return false;
    std::string cmd = ffmpeg_bin + " -y -f lavfi"
        " -i color=c=blue:size=" + std::to_string(w) + "x" + std::to_string(h) +
        ":rate=25 -vframes " + std::to_string(frames) +
        " -vcodec libx264 -pix_fmt yuv420p"
        " \"" + path.string() + "\" >/dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0 ||
        !fs::exists(path) || fs::file_size(path) == 0) {
        fs::remove(path);
        return false;
    }
    return true;
}

} // namespace

// ── Concept check ────────────────────────────────────────────────────────────

TEST(VideoFileCaptureTest, SatisfiesCameraSourceTypeConcept) {
    static_assert(CameraSourceType<VideoFileCapture>);
}

// ── Error paths (no video file needed) ──────────────────────────────────────

TEST(VideoFileCaptureTest, GetFrameWithoutStartReturnsUnavailable) {
    VideoFileCapture cap("/some/path.mp4");
    auto result = cap.getFrame();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, Error::Code::CameraUnavailable);
}

TEST(VideoFileCaptureTest, StartWithNonexistentPathThrows) {
    VideoFileCapture cap("/nonexistent/video.mp4");
    EXPECT_THROW(cap.start(), improc::FileNotFoundError);
}

TEST(VideoFileCaptureTest, StopIsIdempotent) {
    VideoFileCapture cap("/nonexistent.mp4");
    cap.stop();  // must not crash when never started
    cap.stop();
}

TEST(VideoFileCaptureTest, GetFrameAfterStopReturnsUnavailable) {
    auto tmp = fs::temp_directory_path() / "stop_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 4, 32, 24))
        GTEST_SKIP() << "ffmpeg unavailable";

    VideoFileCapture cap(tmp);
    cap.start();
    cap.stop();
    auto result = cap.getFrame();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, Error::Code::CameraUnavailable);
}

// ── Happy path (require ffmpeg) ──────────────────────────────────────────────

TEST(VideoFileCaptureTest, GetFrameReturnsRgbFrame) {
    auto tmp = fs::temp_directory_path() / "cap_frame_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 4, 64, 48))
        GTEST_SKIP() << "ffmpeg unavailable";

    VideoFileCapture cap(tmp);
    cap.start();
    auto result = cap.getFrame();
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_TRUE(result->rgb.has_value());
    EXPECT_EQ(result->rgb->mat().rows, 48);
    EXPECT_EQ(result->rgb->mat().cols, 64);
    EXPECT_FALSE(result->depth.has_value());
    EXPECT_FALSE(result->source_id.empty());
    cap.stop();
}

TEST(VideoFileCaptureTest, ReturnsEndOfFileAfterLastFrame) {
    auto tmp = fs::temp_directory_path() / "eof_cap_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 3, 32, 24))
        GTEST_SKIP() << "ffmpeg unavailable";

    VideoFileCapture cap(tmp);
    cap.start();
    int count = 0;
    while (true) {
        auto f = cap.getFrame();
        if (!f.has_value()) {
            EXPECT_EQ(f.error().code, Error::Code::EndOfFile);
            break;
        }
        ++count;
        ASSERT_LT(count, 100) << "EOF never returned";
    }
    EXPECT_GT(count, 0);
    cap.stop();
}

TEST(VideoFileCaptureTest, WorksWithAnyCameraSource) {
    auto tmp = fs::temp_directory_path() / "any_cap_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 2, 32, 24))
        GTEST_SKIP() << "ffmpeg unavailable";

    auto src = improc::io::AnyCameraSource::make<VideoFileCapture>(tmp);
    EXPECT_TRUE(static_cast<bool>(src));
    src.start();
    auto f = src.getFrame();
    EXPECT_TRUE(f.has_value());
    src.stop();
}
