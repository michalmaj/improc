// tests/views/test_views_m3_video.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <memory>
#include "improc/core/pipeline.hpp"
#include "improc/io/video_reader.hpp"
#include "improc/io/video_writer.hpp"
#include "improc/views/views.hpp"

using namespace improc::core;
using namespace improc::io;
namespace views = improc::views;
namespace fs    = std::filesystem;

class VideoViewTest : public ::testing::Test {
protected:
    static constexpr int kFrames = 10;
    static constexpr int kWidth  = 32;
    static constexpr int kHeight = 32;

    fs::path                        video_path_;
    std::unique_ptr<VideoReader>    reader_;

    void SetUp() override {
        video_path_ = fs::temp_directory_path() / "improc_test_m3_video.avi";
        {
            VideoWriter writer{video_path_};
            writer.fps(25.0);
            for (int i = 0; i < kFrames; ++i) {
                cv::Mat mat(kHeight, kWidth, CV_8UC3, cv::Scalar(i * 20, 100, 200));
                writer(Image<BGR>{mat});
            }
        } // VideoWriter destructor finalises the file
        reader_ = std::make_unique<VideoReader>(video_path_);
    }

    void TearDown() override {
        reader_.reset();
        fs::remove(video_path_);
    }
};

// ── Iteration ─────────────────────────────────────────────────────────────────

TEST_F(VideoViewTest, IterateAllFrames) {
    auto view = views::VideoView{*reader_};
    int count = 0;
    for (const auto& frame : view) {
        EXPECT_EQ(frame.cols(), kWidth);
        EXPECT_EQ(frame.rows(), kHeight);
        ++count;
    }
    EXPECT_EQ(count, kFrames);
}

TEST_F(VideoViewTest, MaterializeToVector) {
    auto frames = views::VideoView{*reader_}
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(static_cast<int>(frames.size()), kFrames);
}

// ── transform ─────────────────────────────────────────────────────────────────

TEST_F(VideoViewTest, TransformResizesAllFrames) {
    auto frames = views::VideoView{*reader_}
        | views::transform(Resize{}.width(16).height(16))
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(static_cast<int>(frames.size()), kFrames);
    for (const auto& f : frames) {
        EXPECT_EQ(f.cols(), 16);
        EXPECT_EQ(f.rows(), 16);
    }
}

// ── take / drop ───────────────────────────────────────────────────────────────

TEST_F(VideoViewTest, TakeFirstN) {
    auto frames = views::VideoView{*reader_}
        | views::take(4)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(frames.size(), 4u);
}

TEST_F(VideoViewTest, TakeMoreThanAvailable) {
    auto frames = views::VideoView{*reader_}
        | views::take(100)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(static_cast<int>(frames.size()), kFrames);
}

TEST_F(VideoViewTest, DropFirstN) {
    auto frames = views::VideoView{*reader_}
        | views::drop(3)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(static_cast<int>(frames.size()), kFrames - 3);
}

TEST_F(VideoViewTest, DropMoreThanAvailable) {
    auto frames = views::VideoView{*reader_}
        | views::drop(100)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_TRUE(frames.empty());
}

// ── filter ────────────────────────────────────────────────────────────────────

TEST_F(VideoViewTest, FilterKeepsMatchingFrames) {
    auto frames = views::VideoView{*reader_}
        | views::filter([](const Image<BGR>& f) { return f.cols() == kWidth; })
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(static_cast<int>(frames.size()), kFrames);
}

TEST_F(VideoViewTest, FilterRejectsAll) {
    auto frames = views::VideoView{*reader_}
        | views::filter([](const Image<BGR>&) { return false; })
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_TRUE(frames.empty());
}

// ── composition ───────────────────────────────────────────────────────────────

TEST_F(VideoViewTest, DropThenTake) {
    auto frames = views::VideoView{*reader_}
        | views::drop(3)
        | views::take(4)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(frames.size(), 4u);
}

TEST_F(VideoViewTest, TransformThenTake) {
    auto frames = views::VideoView{*reader_}
        | views::transform(Resize{}.width(16).height(16))
        | views::take(5)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(frames.size(), 5u);
    for (const auto& f : frames)
        EXPECT_EQ(f.cols(), 16);
}
