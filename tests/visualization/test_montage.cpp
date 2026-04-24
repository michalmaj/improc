// tests/visualization/test_montage.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "improc/visualization/montage.hpp"
#include "improc/exceptions.hpp"

using improc::visualization::Montage;
using improc::core::Image;
using improc::core::BGR;

namespace {

Image<BGR> make_img(int w, int h, cv::Scalar color = {100, 150, 200}) {
    return Image<BGR>(cv::Mat(h, w, CV_8UC3, color));
}

std::vector<Image<BGR>> make_imgs(int n, int w = 32, int h = 32) {
    std::vector<Image<BGR>> v;
    v.reserve(n);
    for (int i = 0; i < n; i++)
        v.push_back(make_img(w, h, cv::Scalar(i * 20 % 256, 100, 200)));
    return v;
}

} // namespace

// ── Error cases ─────────────────────────────────────────────────────────────

TEST(MontageTest, EmptyVectorThrows) {
    std::vector<Image<BGR>> empty;
    EXPECT_THROW(Montage{empty}, improc::ParameterError);
}

TEST(MontageTest, ZeroColsThrows) {
    auto imgs = make_imgs(4);
    EXPECT_THROW(Montage{imgs}.cols(0), improc::ParameterError);
}

TEST(MontageTest, NegativeColsThrows) {
    auto imgs = make_imgs(4);
    EXPECT_THROW(Montage{imgs}.cols(-1), improc::ParameterError);
}

TEST(MontageTest, ZeroCellWidthThrows) {
    auto imgs = make_imgs(4);
    EXPECT_THROW(Montage{imgs}.cell_size(0, 32), improc::ParameterError);
}

TEST(MontageTest, ZeroCellHeightThrows) {
    auto imgs = make_imgs(4);
    EXPECT_THROW(Montage{imgs}.cell_size(32, 0), improc::ParameterError);
}

TEST(MontageTest, NegativeGapThrows) {
    auto imgs = make_imgs(4);
    EXPECT_THROW(Montage{imgs}.gap(-1), improc::ParameterError);
}

// ── Output type ─────────────────────────────────────────────────────────────

TEST(MontageTest, OutputIsImageBGR) {
    auto imgs = make_imgs(4, 32, 32);
    Image<BGR> grid = Montage{imgs}.cols(2).cell_size(32, 32)();
    EXPECT_EQ(grid.mat().type(), CV_8UC3);
}

// ── Dimensions ──────────────────────────────────────────────────────────────

TEST(MontageTest, DimensionsWithExplicitColsNoGap) {
    // 4 images, 2 cols, 2 rows, cell 32x32, gap 0
    // expected: width = 2*32 = 64, height = 2*32 = 64
    auto imgs = make_imgs(4, 32, 32);
    Image<BGR> grid = Montage{imgs}.cols(2).cell_size(32, 32)();
    EXPECT_EQ(grid.cols(), 64);
    EXPECT_EQ(grid.rows(), 64);
}

TEST(MontageTest, DimensionsWithGap) {
    // 4 images, 2 cols, 2 rows, cell 32x32, gap 4
    // width  = 2*32 + 1*4 = 68
    // height = 2*32 + 1*4 = 68
    auto imgs = make_imgs(4, 32, 32);
    Image<BGR> grid = Montage{imgs}.cols(2).cell_size(32, 32).gap(4)();
    EXPECT_EQ(grid.cols(), 68);
    EXPECT_EQ(grid.rows(), 68);
}

TEST(MontageTest, DimensionsWithOddCount) {
    // 5 images, 3 cols → 2 rows (ceil(5/3)), cell 32x32, gap 0
    // width = 3*32 = 96, height = 2*32 = 64
    auto imgs = make_imgs(5, 32, 32);
    Image<BGR> grid = Montage{imgs}.cols(3).cell_size(32, 32)();
    EXPECT_EQ(grid.cols(), 96);
    EXPECT_EQ(grid.rows(), 64);
}

TEST(MontageTest, SingleImage) {
    auto imgs = make_imgs(1, 48, 36);
    Image<BGR> grid = Montage{imgs}();
    EXPECT_EQ(grid.cols(), 48);
    EXPECT_EQ(grid.rows(), 36);
}

// ── Auto-layout ──────────────────────────────────────────────────────────────

TEST(MontageTest, AutoColsForFourImages) {
    // ceil(sqrt(4)) = 2
    auto imgs = make_imgs(4, 32, 32);
    Image<BGR> grid = Montage{imgs}.cell_size(32, 32)();
    EXPECT_EQ(grid.cols(), 64);   // 2 cols * 32
    EXPECT_EQ(grid.rows(), 64);   // 2 rows * 32
}

TEST(MontageTest, AutoColsForNineImages) {
    // ceil(sqrt(9)) = 3
    auto imgs = make_imgs(9, 32, 32);
    Image<BGR> grid = Montage{imgs}.cell_size(32, 32)();
    EXPECT_EQ(grid.cols(), 96);   // 3 * 32
    EXPECT_EQ(grid.rows(), 96);   // 3 * 32
}

TEST(MontageTest, AutoColsForTenImages) {
    // ceil(sqrt(10)) = 4
    auto imgs = make_imgs(10, 32, 32);
    Image<BGR> grid = Montage{imgs}.cell_size(32, 32)();
    EXPECT_EQ(grid.cols(), 128);  // 4 * 32
    EXPECT_EQ(grid.rows(), 96);   // ceil(10/4) = 3 rows * 32
}

// ── Cell content ─────────────────────────────────────────────────────────────

TEST(MontageTest, EmptyCellsFilledWithBackground) {
    // 5 images in a 3x2 grid: last cell (position [1][2]) should be black
    cv::Scalar bg{50, 60, 70};
    auto imgs = make_imgs(5, 32, 32);
    Image<BGR> grid = Montage{imgs}.cols(3).cell_size(32, 32).background(bg)();

    // Bottom-right cell starts at x=64, y=32
    cv::Vec3b pixel = grid.mat().at<cv::Vec3b>(32, 64);
    EXPECT_EQ(pixel[0], 50);
    EXPECT_EQ(pixel[1], 60);
    EXPECT_EQ(pixel[2], 70);
}

TEST(MontageTest, ImagesOfDifferentSizesRescaled) {
    // Mix 32x32 and 64x64 images; cell_size(32,32) should resize all
    std::vector<Image<BGR>> imgs;
    imgs.push_back(make_img(32, 32));
    imgs.push_back(make_img(64, 64));
    Image<BGR> grid = Montage{imgs}.cols(2).cell_size(32, 32)();
    EXPECT_EQ(grid.cols(), 64);
    EXPECT_EQ(grid.rows(), 32);
}
