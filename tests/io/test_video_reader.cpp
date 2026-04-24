// tests/io/test_video_reader.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <string>
#include <opencv2/videoio.hpp>
#include "improc/io/video_reader.hpp"
#include "improc/exceptions.hpp"

using improc::io::VideoReader;
using improc::core::Image;
using improc::core::BGR;
namespace fs = std::filesystem;

namespace {

struct TempFile {
    explicit TempFile(fs::path p) : path(std::move(p)) {}
    ~TempFile() { fs::remove(path); }
    fs::path path;
};

// Creates a small H.264 test video using the system ffmpeg.
// H.264/MP4 is reliably decodable by cv::VideoCapture even in minimal builds.
// Returns false if ffmpeg is unavailable or the output file is unreadable.
bool make_test_video(const fs::path& path, int frames = 8,
                     int w = 64, int h = 48) {
    // Try common ffmpeg locations; PATH may not be fully set in test subprocess.
    const char* ffmpeg_paths[] = {
        "ffmpeg",
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        nullptr
    };
    std::string ffmpeg_bin;
    for (const char** p = ffmpeg_paths; *p; ++p) {
        std::string probe = std::string(*p) + " -version >/dev/null 2>&1";
        if (std::system(probe.c_str()) == 0) { ffmpeg_bin = *p; break; }
    }
    if (ffmpeg_bin.empty()) return false;

    // Use H.264 encoded into MP4 — universally decodable by cv::VideoCapture.
    std::string cmd = ffmpeg_bin + " -y -f lavfi"
        " -i color=c=blue:size=" + std::to_string(w) + "x" + std::to_string(h) +
        ":rate=25"
        " -vframes " + std::to_string(frames) +
        " -vcodec libx264 -pix_fmt yuv420p"
        " \"" + path.string() + "\""
        " >/dev/null 2>&1";
    if (std::system(cmd.c_str()) != 0 ||
        !fs::exists(path) || fs::file_size(path) == 0) {
        fs::remove(path);
        return false;
    }
    return true;
}

} // namespace

// ── Constructor errors ──────────────────────────────────────────────────────

TEST(VideoReaderTest, NonexistentPathThrows) {
    EXPECT_THROW(VideoReader{"/nonexistent/path/video.avi"},
                 improc::FileNotFoundError);
}

TEST(VideoReaderTest, NonVideoFileThrows) {
    auto tmp = fs::temp_directory_path() / "not_a_video.txt";
    { std::ofstream f(tmp); f << "hello"; }
    TempFile guard{tmp};
    EXPECT_THROW(VideoReader{tmp}, improc::IoError);
}

// ── Metadata accessors ──────────────────────────────────────────────────────

TEST(VideoReaderTest, MetadataAfterOpen) {
    auto tmp = fs::temp_directory_path() / "meta_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 8, 64, 48))
        GTEST_SKIP() << "video creation unavailable (no ffmpeg or H.264 encoder)";

    VideoReader reader{tmp};
    EXPECT_GT(reader.fps(), 0.0);
    EXPECT_EQ(reader.width(),  64);
    EXPECT_EQ(reader.height(), 48);
    EXPECT_TRUE(reader.is_open());
}

// ── Frame iteration ─────────────────────────────────────────────────────────

TEST(VideoReaderTest, ReturnsCorrectNumberOfFrames) {
    auto tmp = fs::temp_directory_path() / "count_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 8, 64, 48))
        GTEST_SKIP() << "video creation unavailable (no ffmpeg or H.264 encoder)";

    VideoReader reader{tmp};
    int count = 0;
    while (auto f = reader.next()) ++count;
    EXPECT_EQ(count, 8);
}

TEST(VideoReaderTest, FrameDimensionsMatchVideo) {
    auto tmp = fs::temp_directory_path() / "dim_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 4, 64, 48))
        GTEST_SKIP() << "video creation unavailable (no ffmpeg or H.264 encoder)";

    VideoReader reader{tmp};
    auto frame = reader.next();
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->cols(), 64);
    EXPECT_EQ(frame->rows(), 48);
}

TEST(VideoReaderTest, FrameIsImageBGR) {
    auto tmp = fs::temp_directory_path() / "type_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 2, 64, 48))
        GTEST_SKIP() << "video creation unavailable (no ffmpeg or H.264 encoder)";

    VideoReader reader{tmp};
    auto frame = reader.next();
    ASSERT_TRUE(frame.has_value());
    EXPECT_EQ(frame->mat().type(), CV_8UC3);
}

TEST(VideoReaderTest, NextAfterEOFReturnsNullopt) {
    auto tmp = fs::temp_directory_path() / "eof_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 2, 32, 24))
        GTEST_SKIP() << "video creation unavailable (no ffmpeg or H.264 encoder)";

    VideoReader reader{tmp};
    while (reader.next()) {}             // drain
    EXPECT_FALSE(reader.next().has_value()); // idempotent
    EXPECT_FALSE(reader.next().has_value());
}

// ── Close ───────────────────────────────────────────────────────────────────

TEST(VideoReaderTest, CloseMarksAsNotOpen) {
    auto tmp = fs::temp_directory_path() / "close_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 2, 32, 24))
        GTEST_SKIP() << "video creation unavailable (no ffmpeg or H.264 encoder)";

    VideoReader reader{tmp};
    EXPECT_TRUE(reader.is_open());
    reader.close();
    EXPECT_FALSE(reader.is_open());
}

TEST(VideoReaderTest, NextAfterCloseReturnsNullopt) {
    auto tmp = fs::temp_directory_path() / "close_next_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 4, 32, 24))
        GTEST_SKIP() << "video creation unavailable (no ffmpeg or H.264 encoder)";

    VideoReader reader{tmp};
    reader.close();
    EXPECT_FALSE(reader.next().has_value());
}

TEST(VideoReaderTest, FrameCountNormalisedToMinusOneWhenUnavailable) {
    // frame_count() must return either a positive count (for containers that
    // report it reliably) or -1 (never 0) — the normalisation in the
    // implementation converts 0.0 from OpenCV to -1.
    auto tmp = fs::temp_directory_path() / "fcount_test.mp4";
    TempFile guard{tmp};
    if (!make_test_video(tmp, 8, 32, 24))
        GTEST_SKIP() << "video creation unavailable (no ffmpeg or H.264 encoder)";

    VideoReader reader{tmp};
    const int fc = reader.frame_count();
    EXPECT_TRUE(fc > 0 || fc == -1) << "frame_count() returned 0, expected >0 or -1";
}
