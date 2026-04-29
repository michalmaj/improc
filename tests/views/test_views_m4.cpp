// tests/views/test_views_m4.cpp
#include <gtest/gtest.h>
#include <vector>
#include "improc/core/pipeline.hpp"
#include "improc/views/views.hpp"

using namespace improc::core;
namespace views = improc::views;

// ── helpers ──────────────────────────────────────────────────────────────────

static Image<BGR> make_bgr(int w, int h, cv::Scalar color = {100, 150, 200}) {
    return Image<BGR>(cv::Mat(h, w, CV_8UC3, color));
}
static std::vector<Image<BGR>> make_batch(int n, int w = 64, int h = 64) {
    std::vector<Image<BGR>> v;
    for (int i = 0; i < n; ++i)
        v.push_back(make_bgr(w, h, cv::Scalar(i * 20, 100, 200)));
    return v;
}

// ── views::batch ─────────────────────────────────────────────────────────────

TEST(ViewsM4, BatchSplitsEvenly) {
    auto imgs = make_batch(6);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs | views::batch(3))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 2u);
    EXPECT_EQ(batches[0].size(), 3u);
    EXPECT_EQ(batches[1].size(), 3u);
}

TEST(ViewsM4, BatchLastChunkSmaller) {
    auto imgs = make_batch(5);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs | views::batch(2))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 3u);
    EXPECT_EQ(batches[0].size(), 2u);
    EXPECT_EQ(batches[1].size(), 2u);
    EXPECT_EQ(batches[2].size(), 1u);
}

TEST(ViewsM4, BatchLargerThanSource) {
    auto imgs = make_batch(3);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs | views::batch(10))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 1u);
    EXPECT_EQ(batches[0].size(), 3u);
}

TEST(ViewsM4, BatchEmptySource) {
    std::vector<Image<BGR>> empty;
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : empty | views::batch(4))
        batches.push_back(b);
    EXPECT_TRUE(batches.empty());
}

TEST(ViewsM4, BatchSourceUnchanged) {
    auto imgs = make_batch(4);
    for (const auto& b : imgs | views::batch(2)) (void)b;
    EXPECT_EQ(imgs.size(), 4u);
    EXPECT_EQ(imgs[0].cols(), 64);
}

TEST(ViewsM4, BatchAfterTransform) {
    auto imgs = make_batch(6);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs
            | views::transform(Resize{}.width(32).height(32))
            | views::batch(2))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 3u);
    for (const auto& batch : batches)
        for (const auto& img : batch)
            EXPECT_EQ(img.cols(), 32);
}

TEST(ViewsM4, BatchAfterFilter) {
    auto imgs = make_batch(6);
    for (int i = 0; i < 3; ++i)
        imgs[i] = make_bgr(128, 128);
    // 3 large images → batch(2) → [2, 1]
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs
            | views::filter([](const Image<BGR>& img) { return img.cols() == 128; })
            | views::batch(2))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 2u);
    EXPECT_EQ(batches[0].size(), 2u);
    EXPECT_EQ(batches[1].size(), 1u);
}

TEST(ViewsM4, BatchAfterTake) {
    auto imgs = make_batch(10);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs | views::take(6) | views::batch(3))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 2u);
    EXPECT_EQ(batches[0].size(), 3u);
    EXPECT_EQ(batches[1].size(), 3u);
}

TEST(ViewsM4, BatchAfterDrop) {
    auto imgs = make_batch(8);
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : imgs | views::drop(4) | views::batch(2))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 2u);
    for (const auto& batch : batches)
        EXPECT_EQ(batch.size(), 2u);
}

// ── batch: DirView source ─────────────────────────────────────────────────────

#include <filesystem>
#include <format>
namespace fs = std::filesystem;

class ViewsM4DirBatch : public ::testing::Test {
protected:
    fs::path dir_;
    void SetUp() override {
        dir_ = fs::temp_directory_path() / "test_m4_dir_batch";
        fs::create_directories(dir_);
        for (int i = 0; i < 6; ++i) {
            cv::Mat m(64, 64, CV_8UC3, cv::Scalar(i * 20, 100, 200));
            cv::imwrite((dir_ / std::format("{:04d}.png", i)).string(), m);
        }
    }
    void TearDown() override { fs::remove_all(dir_); }
};

TEST_F(ViewsM4DirBatch, DirViewBatchSplitsEvenly) {
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : views::from_dir(dir_, {".png"}) | views::batch(2))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 3u);
    for (const auto& batch : batches)
        EXPECT_EQ(batch.size(), 2u);
}

TEST_F(ViewsM4DirBatch, DirViewBatchLastChunkSmaller) {
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : views::from_dir(dir_, {".png"}) | views::batch(4))
        batches.push_back(b);
    ASSERT_EQ(batches.size(), 2u);
    EXPECT_EQ(batches[0].size(), 4u);
    EXPECT_EQ(batches[1].size(), 2u);
}

// ── batch: VideoView source ───────────────────────────────────────────────────

#include "improc/io/video_reader.hpp"
using improc::io::VideoReader;

static bool ffmpeg_ok() {
    return std::system("ffmpeg -version > /dev/null 2>&1") == 0;
}

static std::string make_video(const fs::path& dir, int n) {
    auto path   = dir / "test.mp4";
    auto frames = dir / "frames";
    fs::create_directories(frames);
    for (int i = 0; i < n; ++i) {
        cv::Mat m(64, 64, CV_8UC3, cv::Scalar(i * 10, 100, 200));
        cv::imwrite((frames / std::format("{:04d}.png", i)).string(), m);
    }
    std::system(std::format(
        "ffmpeg -y -framerate 10 -i {}/%04d.png -c:v libx264 -pix_fmt yuv420p {} > /dev/null 2>&1",
        frames.string(), path.string()).c_str());
    fs::remove_all(frames);
    return path.string();
}

class ViewsM4VideoBatch : public ::testing::Test {
protected:
    fs::path    dir_;
    std::string video_path_;
    void SetUp() override {
        dir_ = fs::temp_directory_path() / "test_m4_video_batch";
        fs::create_directories(dir_);
        if (!ffmpeg_ok()) return;
        video_path_ = make_video(dir_, 6);
    }
    void TearDown() override { fs::remove_all(dir_); }
};

TEST_F(ViewsM4VideoBatch, VideoViewBatchSplitsEvenly) {
    if (!ffmpeg_ok() || !fs::exists(video_path_)) GTEST_SKIP();
    VideoReader reader{video_path_};
    std::vector<std::vector<Image<BGR>>> batches;
    for (const auto& b : views::VideoView{reader} | views::batch(2))
        batches.push_back(b);
    EXPECT_EQ(batches.size(), 3u);
    for (const auto& batch : batches)
        EXPECT_EQ(batch.size(), 2u);
}

// ── views::enumerate ─────────────────────────────────────────────────────────

TEST(ViewsM4, EnumerateStartsAtZero) {
    auto imgs = make_batch(3);
    std::size_t first_idx = 999;
    for (const auto& [idx, img] : imgs | views::enumerate) {
        first_idx = idx;
        break;
    }
    EXPECT_EQ(first_idx, 0u);
}

TEST(ViewsM4, EnumerateIncrementsIndex) {
    auto imgs = make_batch(4);
    std::vector<std::size_t> indices;
    for (const auto& [idx, img] : imgs | views::enumerate)
        indices.push_back(idx);
    ASSERT_EQ(indices.size(), 4u);
    for (std::size_t i = 0; i < indices.size(); ++i)
        EXPECT_EQ(indices[i], i);
}

TEST(ViewsM4, EnumeratePreservesImages) {
    auto imgs = make_batch(3, 64, 64);
    for (const auto& [idx, img] : imgs | views::enumerate) {
        EXPECT_EQ(img.cols(), 64);
        EXPECT_EQ(img.rows(), 64);
    }
}

TEST(ViewsM4, EnumerateEmptySource) {
    std::vector<Image<BGR>> empty;
    int count = 0;
    for (const auto& [idx, img] : empty | views::enumerate)
        ++count;
    EXPECT_EQ(count, 0);
}

TEST(ViewsM4, EnumerateAfterTransform) {
    auto imgs = make_batch(3);
    std::vector<std::size_t> indices;
    for (const auto& [idx, img] : imgs
            | views::transform(Resize{}.width(32).height(32))
            | views::enumerate) {
        indices.push_back(idx);
        EXPECT_EQ(img.cols(), 32);
    }
    EXPECT_EQ(indices.size(), 3u);
}

TEST(ViewsM4, EnumerateAfterFilter) {
    auto imgs = make_batch(6);
    for (int i = 0; i < 3; ++i)
        imgs[i] = make_bgr(128, 128);
    std::vector<std::size_t> indices;
    for (const auto& [idx, img] : imgs
            | views::filter([](const Image<BGR>& img) { return img.cols() == 128; })
            | views::enumerate)
        indices.push_back(idx);
    // enumerate resets to 0 for the filtered view (not original source indices)
    ASSERT_EQ(indices.size(), 3u);
    EXPECT_EQ(indices[0], 0u);
    EXPECT_EQ(indices[2], 2u);
}

TEST(ViewsM4, EnumerateAfterTake) {
    auto imgs = make_batch(8);
    std::vector<std::size_t> indices;
    for (const auto& [idx, img] : imgs | views::take(4) | views::enumerate)
        indices.push_back(idx);
    ASSERT_EQ(indices.size(), 4u);
    EXPECT_EQ(indices.back(), 3u);
}

TEST(ViewsM4, EnumerateAfterDrop) {
    auto imgs = make_batch(8);
    std::vector<std::size_t> indices;
    for (const auto& [idx, img] : imgs | views::drop(5) | views::enumerate)
        indices.push_back(idx);
    ASSERT_EQ(indices.size(), 3u);
    EXPECT_EQ(indices[0], 0u);
}

// ── enumerate: DirView source ─────────────────────────────────────────────────

TEST_F(ViewsM4DirBatch, DirViewEnumerateIndices) {
    std::vector<std::size_t> indices;
    for (const auto& [idx, img] : views::from_dir(dir_, {".png"}) | views::enumerate)
        indices.push_back(idx);
    ASSERT_EQ(indices.size(), 6u);
    for (std::size_t i = 0; i < indices.size(); ++i)
        EXPECT_EQ(indices[i], i);
}

TEST_F(ViewsM4DirBatch, DirViewEnumeratePreservesImage) {
    for (const auto& [idx, img] : views::from_dir(dir_, {".png"}) | views::enumerate) {
        EXPECT_EQ(img.cols(), 64);
        EXPECT_EQ(img.rows(), 64);
    }
}

// ── enumerate: VideoView source ───────────────────────────────────────────────

TEST_F(ViewsM4VideoBatch, VideoViewEnumerateIndices) {
    if (!ffmpeg_ok() || !fs::exists(video_path_)) GTEST_SKIP();
    VideoReader reader{video_path_};
    std::vector<std::size_t> indices;
    for (const auto& [idx, img] : views::VideoView{reader} | views::enumerate)
        indices.push_back(idx);
    ASSERT_EQ(indices.size(), 6u);
    for (std::size_t i = 0; i < indices.size(); ++i)
        EXPECT_EQ(indices[i], i);
}
