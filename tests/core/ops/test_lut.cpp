// tests/core/ops/test_lut.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

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
    EXPECT_THROW(LUT{bad}, improc::ParameterError);
}

TEST(LUTTest, InvalidTableDepthThrows) {
    cv::Mat bad(1, 256, CV_16UC1, cv::Scalar(0));
    EXPECT_THROW(LUT{bad}, improc::ParameterError);
}

TEST(LUTTest, InvertsGrayImageValues) {
    Image<Gray> img = make_gray(4, 4, 100);
    Image<Gray> result = img | LUT{make_invert_table()};
    EXPECT_EQ(result.mat().at<uint8_t>(0, 0), 155);  // 255 - 100
}

TEST(LUTTest, ReturnsSameSizeAsInput) {
    Image<Gray> result = make_gray(16, 32) | LUT{make_invert_table()};
    EXPECT_EQ(result.rows(), 16);
    EXPECT_EQ(result.cols(), 32);
}

TEST(LUTTest, WorksOnBGRImage) {
    // Build halving table: output[i] = i / 2
    cv::Mat half_table(1, 256, CV_8UC1);
    for (int i = 0; i < 256; ++i)
        half_table.at<uchar>(i) = static_cast<uchar>(i / 2);

    LUT op(half_table);
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    auto result = op(img);

    ASSERT_EQ(result.mat().size(), img.mat().size());
    auto px = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_EQ(px[0], 50);   // 100 / 2
    EXPECT_EQ(px[1], 75);   // 150 / 2
    EXPECT_EQ(px[2], 100);  // 200 / 2
}
