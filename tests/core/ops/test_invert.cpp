// tests/core/ops/test_invert.cpp
#include <gtest/gtest.h>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

static Image<Gray> make_gray(int rows, int cols, uchar value) {
    cv::Mat mat(rows, cols, CV_8UC1, cv::Scalar(value));
    return Image<Gray>{mat};
}
static Image<BGR> make_bgr(int rows, int cols, uchar b, uchar g, uchar r) {
    cv::Mat mat(rows, cols, CV_8UC3, cv::Scalar(b, g, r));
    return Image<BGR>{mat};
}

TEST(InvertTest, InvertGrayFlipsWhiteToBlack) {
    Image<Gray> img = make_gray(10, 10, 255);
    Image<Gray> result = img | Invert{};
    EXPECT_EQ(cv::countNonZero(result.mat()), 0);
}

TEST(InvertTest, InvertGrayFlipsBlackToWhite) {
    Image<Gray> img = make_gray(10, 10, 0);
    Image<Gray> result = img | Invert{};
    EXPECT_EQ(cv::countNonZero(result.mat()), 100);
}

TEST(InvertTest, InvertBGRFlipsChannels) {
    // B=50 → 205, G=100 → 155, R=150 → 105
    Image<BGR> img = make_bgr(10, 10, 50, 100, 150);
    Image<BGR> result = img | Invert{};
    cv::Vec3b px = result.mat().at<cv::Vec3b>(5, 5);
    EXPECT_EQ(px[0], 205);
    EXPECT_EQ(px[1], 155);
    EXPECT_EQ(px[2], 105);
}

TEST(InvertTest, InvertPreservesSize) {
    Image<Gray> img = make_gray(20, 30, 128);
    Image<Gray> result = img | Invert{};
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 30);
}

TEST(InvertTest, InvertGrayPreservesType) {
    Image<Gray> img = make_gray(10, 10, 128);
    Image<Gray> result = img | Invert{};
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(InvertTest, InvertBGRPreservesType) {
    Image<BGR> img = make_bgr(10, 10, 100, 100, 100);
    Image<BGR> result = img | Invert{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(InvertTest, InvertTwiceIsIdentity) {
    Image<Gray> img = make_gray(10, 10, 128);
    Image<Gray> result = img | Invert{} | Invert{};
    cv::Mat diff;
    cv::absdiff(img.mat(), result.mat(), diff);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

TEST(InvertTest, PipelineSyntaxWorks) {
    Image<Gray> img = make_gray(10, 10, 200);
    Image<Gray> result = img | Invert{};
    EXPECT_EQ(result.rows(), 10);
}
