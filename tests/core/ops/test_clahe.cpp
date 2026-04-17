// tests/core/ops/test_clahe.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/core/ops/clahe.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// ── Parameter validation ─────────────────────────────────────────────────────

TEST(CLAHETest, ZeroClipLimitThrows) {
    EXPECT_THROW(CLAHE{}.clip_limit(0.0), improc::ParameterError);
}

TEST(CLAHETest, NegativeClipLimitThrows) {
    EXPECT_THROW(CLAHE{}.clip_limit(-1.0), improc::ParameterError);
}

TEST(CLAHETest, ZeroTileWidthThrows) {
    EXPECT_THROW(CLAHE{}.tile_grid_size(0, 8), improc::ParameterError);
}

TEST(CLAHETest, ZeroTileHeightThrows) {
    EXPECT_THROW(CLAHE{}.tile_grid_size(8, 0), improc::ParameterError);
}

// ── Gray output ──────────────────────────────────────────────────────────────

TEST(CLAHETest, GrayPreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    auto result = CLAHE{}(img);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(CLAHETest, GrayPipelineForm) {
    cv::Mat mat(32, 32, CV_8UC1, cv::Scalar(80));
    Image<Gray> img(mat);
    auto result = img | CLAHE{}.clip_limit(2.0).tile_grid_size(4, 4);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}

TEST(CLAHETest, GrayEnhancesContrast) {
    // Build a low-contrast gradient and verify output has wider spread
    cv::Mat mat(64, 64, CV_8UC1);
    for (int r = 0; r < 64; ++r)
        for (int c = 0; c < 64; ++c)
            mat.at<uchar>(r, c) = static_cast<uchar>(100 + (r + c) / 4);  // range ~[100,131]

    Image<Gray> img(mat);
    auto result = CLAHE{}.clip_limit(2.0)(img);

    double in_min, in_max, out_min, out_max;
    cv::minMaxLoc(mat, &in_min, &in_max);
    cv::minMaxLoc(result.mat(), &out_min, &out_max);
    EXPECT_GT(out_max - out_min, in_max - in_min);
}

// ── BGR output ───────────────────────────────────────────────────────────────

TEST(CLAHETest, BGRPreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(80, 100, 120));
    Image<BGR> img(mat);
    auto result = CLAHE{}(img);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(CLAHETest, BGRPipelineForm) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(60, 80, 100));
    Image<BGR> img(mat);
    auto result = img | CLAHE{}.clip_limit(3.0).tile_grid_size(8, 8);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}

TEST(CLAHETest, BGRDoesNotThrowOnValidInput) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(50, 100, 150));
    Image<BGR> img(mat);
    EXPECT_NO_THROW(CLAHE{}.clip_limit(2.0)(img));
}
