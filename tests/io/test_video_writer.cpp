// tests/io/test_video_writer.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include "improc/io/video_writer.hpp"
#include "improc/exceptions.hpp"

using improc::io::VideoWriter;
using improc::core::Image;
using improc::core::BGR;
namespace fs = std::filesystem;

namespace {

// RAII helper: deletes a file on scope exit.
struct TempFile {
    explicit TempFile(fs::path p) : path(std::move(p)) {}
    ~TempFile() { fs::remove(path); }
    fs::path path;
};

Image<BGR> make_frame(int w = 64, int h = 48) {
    return Image<BGR>(cv::Mat(h, w, CV_8UC3, cv::Scalar(100, 150, 200)));
}

} // namespace

// ── Parameter validation ────────────────────────────────────────────────────

TEST(VideoWriterTest, NegativeFpsThrows) {
    VideoWriter w{fs::temp_directory_path() / "dummy.mp4"};
    EXPECT_THROW(w.fps(-1.0), improc::ParameterError);
}

TEST(VideoWriterTest, ZeroFpsThrows) {
    VideoWriter w{fs::temp_directory_path() / "dummy.mp4"};
    EXPECT_THROW(w.fps(0.0), improc::ParameterError);
}

TEST(VideoWriterTest, ZeroWidthThrows) {
    VideoWriter w{fs::temp_directory_path() / "dummy.mp4"};
    EXPECT_THROW(w.size(0, 480), improc::ParameterError);
}

TEST(VideoWriterTest, ZeroHeightThrows) {
    VideoWriter w{fs::temp_directory_path() / "dummy.mp4"};
    EXPECT_THROW(w.size(640, 0), improc::ParameterError);
}

TEST(VideoWriterTest, ShortCodecThrows) {
    VideoWriter w{fs::temp_directory_path() / "dummy.mp4"};
    EXPECT_THROW(w.codec("mp4"), improc::ParameterError);
}

TEST(VideoWriterTest, LongCodecThrows) {
    VideoWriter w{fs::temp_directory_path() / "dummy.mp4"};
    EXPECT_THROW(w.codec("mp4vx"), improc::ParameterError);
}

// ── Basic write ─────────────────────────────────────────────────────────────

TEST(VideoWriterTest, WritesToMp4AndCreatesFile) {
    auto path = fs::temp_directory_path() / "improc_test_video.mp4";
    TempFile guard{path};

    {
        VideoWriter w{path};
        w.fps(25);
        auto frame = make_frame();
        for (int i = 0; i < 10; ++i) w(frame);
    } // destructor closes file

    EXPECT_TRUE(fs::exists(path));
    EXPECT_GT(fs::file_size(path), 0u);
}

TEST(VideoWriterTest, WritesToAvi) {
    auto path = fs::temp_directory_path() / "improc_test_video.avi";
    TempFile guard{path};

    {
        VideoWriter w{path};
        w.fps(25);
        auto frame = make_frame();
        for (int i = 0; i < 5; ++i) w(frame);
    }

    EXPECT_TRUE(fs::exists(path));
    EXPECT_GT(fs::file_size(path), 0u);
}

// ── Return value passthrough (pipeline support) ──────────────────────────────

TEST(VideoWriterTest, ReturnsFrameUnchanged) {
    auto path = fs::temp_directory_path() / "improc_test_passthrough.mp4";
    TempFile guard{path};

    VideoWriter w{path};
    w.fps(25);
    auto frame = make_frame(64, 48);
    auto result = w(frame);
    EXPECT_EQ(result.rows(), frame.rows());
    EXPECT_EQ(result.cols(), frame.cols());
}

// ── Auto size detection ─────────────────────────────────────────────────────

TEST(VideoWriterTest, AutoDetectsSizeFromFirstFrame) {
    auto path = fs::temp_directory_path() / "improc_test_autosize.mp4";
    TempFile guard{path};

    VideoWriter w{path};
    w.fps(25);
    EXPECT_FALSE(w.is_open());

    auto frame = make_frame(80, 60);
    w(frame);
    EXPECT_TRUE(w.is_open());
}

// ── Size mismatch ────────────────────────────────────────────────────────────

TEST(VideoWriterTest, FrameSizeMismatchThrows) {
    auto path = fs::temp_directory_path() / "improc_test_sizemismatch.mp4";
    TempFile guard{path};

    VideoWriter w{path};
    w.fps(25).size(64, 48);
    auto first = make_frame(64, 48);
    w(first);  // opens writer at 64x48

    auto wrong = make_frame(128, 96);
    EXPECT_THROW(w(wrong), improc::IoError);
}

// ── close() is idempotent ────────────────────────────────────────────────────

TEST(VideoWriterTest, CloseIsIdempotent) {
    auto path = fs::temp_directory_path() / "improc_test_close.mp4";
    TempFile guard{path};

    VideoWriter w{path};
    w.fps(25);
    w(make_frame());
    EXPECT_NO_THROW(w.close());
    EXPECT_NO_THROW(w.close());
}
