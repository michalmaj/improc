// tests/core/test_lut.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/lut.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

namespace {
cv::Mat make_invert_table() {
    cv::Mat t(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i)
        t.at<uint8_t>(i) = static_cast<uint8_t>(255 - i);
    return t;
}
Image<Gray> make_gray(int rows, int cols, uint8_t val = 100) {
    return Image<Gray>(cv::Mat(rows, cols, CV_8UC1, cv::Scalar(val)));
}
Image<BGR> make_bgr(int rows, int cols) {
    return Image<BGR>(cv::Mat(rows, cols, CV_8UC3, cv::Scalar(50, 100, 150)));
}
} // namespace

TEST(LUTTest, DefaultConstructionFromValidTable) {
    EXPECT_NO_THROW(LUT{make_invert_table()});
}

TEST(LUTTest, InvalidTableSizeThrows) {
    cv::Mat bad(1, 128, CV_8UC1, cv::Scalar(0));
    EXPECT_THROW(LUT{bad}, std::invalid_argument);
}

TEST(LUTTest, InvalidTableDepthThrows) {
    cv::Mat bad(1, 256, CV_16UC1, cv::Scalar(0));
    EXPECT_THROW(LUT{bad}, std::invalid_argument);
}

TEST(LUTTest, InvertsGrayImageValues) {
    Image<Gray> img = make_gray(4, 4, 100);
    Image<Gray> result = img | LUT{make_invert_table()};
    EXPECT_EQ(result.mat().at<uint8_t>(0, 0), 155);  // 255 - 100
}

TEST(LUTTest, ReturnsSameSizeAsInput) {
    Image<Gray> result = make_gray(16, 32) | LUT{make_invert_table()};
    EXPECT_EQ(result.mat().rows, 16);
    EXPECT_EQ(result.mat().cols, 32);
}

TEST(LUTTest, WorksOnBGRImage) {
    Image<BGR> img = make_bgr(8, 8);
    cv::Mat table(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i)
        table.at<uint8_t>(i) = static_cast<uint8_t>(i / 2);
    Image<BGR> result = img | LUT{table};
    EXPECT_EQ(result.mat().rows, 8);
    EXPECT_EQ(result.mat().cols, 8);
}
