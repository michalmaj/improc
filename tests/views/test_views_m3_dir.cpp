// tests/views/test_views_m3_dir.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"
#include "improc/views/views.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;
using namespace improc::io;
namespace views = improc::views;
namespace fs    = std::filesystem;

class DirViewTest : public ::testing::Test {
protected:
    static constexpr int kFiles  = 6;
    static constexpr int kWidth  = 32;
    static constexpr int kHeight = 32;

    fs::path dir_;

    void SetUp() override {
        dir_ = fs::temp_directory_path() / "improc_test_m3_dir";
        fs::create_directories(dir_);

        for (int i = 0; i < kFiles; ++i) {
            cv::Mat mat(kHeight, kWidth, CV_8UC3, cv::Scalar(i * 30, 100, 200));
            auto path = (dir_ / ("img_" + std::to_string(i) + ".png")).string();
            imwrite(path, Image<BGR>{mat});
        }
        // Non-image file — must be ignored by extension filter
        std::ofstream{(dir_ / "readme.txt").string()} << "not an image";
    }

    void TearDown() override {
        fs::remove_all(dir_);
    }
};

// ── Basic iteration ───────────────────────────────────────────────────────────

TEST_F(DirViewTest, IteratesAllPNGs) {
    int count = 0;
    for (const auto& img : views::from_dir(dir_, {".png"})) {
        EXPECT_EQ(img.cols(), kWidth);
        EXPECT_EQ(img.rows(), kHeight);
        ++count;
    }
    EXPECT_EQ(count, kFiles);
}

TEST_F(DirViewTest, IgnoresNonMatchingExtensions) {
    auto result = views::from_dir(dir_, {".jpg"})
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_TRUE(result.empty());
}

TEST_F(DirViewTest, MaterializeToVector) {
    auto imgs = views::from_dir(dir_, {".png"})
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(static_cast<int>(imgs.size()), kFiles);
}

// ── Error handling ────────────────────────────────────────────────────────────

TEST_F(DirViewTest, ThrowsOnMissingDirectory) {
    EXPECT_THROW(
        views::from_dir(dir_ / "nonexistent", {".png"}),
        improc::FileNotFoundError
    );
}

TEST_F(DirViewTest, EmptyDirectoryReturnsEmpty) {
    auto empty_dir = fs::temp_directory_path() / "improc_test_m3_empty";
    fs::create_directories(empty_dir);
    auto result = views::from_dir(empty_dir, {".png"})
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_TRUE(result.empty());
    fs::remove_all(empty_dir);
}

// ── transform ─────────────────────────────────────────────────────────────────

TEST_F(DirViewTest, TransformResizesAll) {
    auto imgs = views::from_dir(dir_, {".png"})
        | views::transform(Resize{}.width(16).height(16))
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(static_cast<int>(imgs.size()), kFiles);
    for (const auto& img : imgs) {
        EXPECT_EQ(img.cols(), 16);
        EXPECT_EQ(img.rows(), 16);
    }
}

// ── filter ────────────────────────────────────────────────────────────────────

TEST_F(DirViewTest, FilterKeepsMatching) {
    auto imgs = views::from_dir(dir_, {".png"})
        | views::filter([](const Image<BGR>& img) { return img.cols() == kWidth; })
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(static_cast<int>(imgs.size()), kFiles);
}

TEST_F(DirViewTest, FilterRejectsAll) {
    auto imgs = views::from_dir(dir_, {".png"})
        | views::filter([](const Image<BGR>&) { return false; })
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_TRUE(imgs.empty());
}

// ── take / drop ───────────────────────────────────────────────────────────────

TEST_F(DirViewTest, TakeFirstN) {
    auto imgs = views::from_dir(dir_, {".png"})
        | views::take(3)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(imgs.size(), 3u);
}

TEST_F(DirViewTest, DropFirstN) {
    auto imgs = views::from_dir(dir_, {".png"})
        | views::drop(2)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(static_cast<int>(imgs.size()), kFiles - 2);
}

TEST_F(DirViewTest, DropMoreThanAvailable) {
    auto imgs = views::from_dir(dir_, {".png"})
        | views::drop(100)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_TRUE(imgs.empty());
}

// ── composition ───────────────────────────────────────────────────────────────

TEST_F(DirViewTest, TransformFilterTake) {
    auto imgs = views::from_dir(dir_, {".png"})
        | views::transform(Resize{}.width(16).height(16))
        | views::filter([](const Image<BGR>& img) { return img.cols() == 16; })
        | views::take(3)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(imgs.size(), 3u);
    for (const auto& img : imgs)
        EXPECT_EQ(img.cols(), 16);
}

TEST_F(DirViewTest, DropThenTake) {
    auto imgs = views::from_dir(dir_, {".png"})
        | views::drop(2)
        | views::take(3)
        | views::to<std::vector<Image<BGR>>>();
    EXPECT_EQ(imgs.size(), 3u);
}
