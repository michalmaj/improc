// tests/core/ops/test_in_range.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
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

TEST(InRangeTest, GrayPixelsInRangeAre255) {
    Image<Gray> img = make_gray(10, 10, 128);
    Image<Gray> result = img | InRange{}.lower({100}).upper({200});
    EXPECT_EQ(cv::countNonZero(result.mat()), 100);
}

TEST(InRangeTest, GrayPixelsOutOfRangeAreZero) {
    Image<Gray> img = make_gray(10, 10, 50);
    Image<Gray> result = img | InRange{}.lower({100}).upper({200});
    EXPECT_EQ(cv::countNonZero(result.mat()), 0);
}

TEST(InRangeTest, OutputIsAlwaysGray) {
    Image<BGR> img = make_bgr(10, 10, 100, 100, 100);
    Image<Gray> result = img | InRange{}.lower({50, 50, 50}).upper({150, 150, 150});
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(InRangeTest, OutputSizeMatchesInput) {
    Image<Gray> img = make_gray(20, 30, 128);
    Image<Gray> result = img | InRange{}.lower({0}).upper({255});
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 30);
}

TEST(InRangeTest, BGRInRangeWorks) {
    // B=100, G=100, R=100 — all channels in [50, 150] → all 100 pixels in range
    Image<BGR> img = make_bgr(10, 10, 100, 100, 100);
    Image<Gray> result = img | InRange{}.lower({50, 50, 50}).upper({150, 150, 150});
    EXPECT_EQ(cv::countNonZero(result.mat()), 100);
}

TEST(InRangeTest, FullRangeAllPixelsIncluded) {
    Image<Gray> img = make_gray(10, 10, 128);
    Image<Gray> result = img | InRange{}.lower({0}).upper({255});
    EXPECT_EQ(cv::countNonZero(result.mat()), 100);
}

TEST(InRangeTest, ThrowsWhenLowerNotSet) {
    Image<Gray> img = make_gray(10, 10, 128);
    InRange op{};
    op.upper({200});
    EXPECT_THROW(op(img), improc::ParameterError);
}

TEST(InRangeTest, ThrowsWhenUpperNotSet) {
    Image<Gray> img = make_gray(10, 10, 128);
    InRange op{};
    op.lower({100});
    EXPECT_THROW(op(img), improc::ParameterError);
}

TEST(InRangeTest, PipelineSyntaxWorks) {
    // 128 in [0, 255] → all 100 pixels should be 255
    Image<Gray> img = make_gray(10, 10, 128);
    Image<Gray> result = img | InRange{}.lower({0}).upper({255});
    EXPECT_EQ(result.mat().at<uchar>(5, 5), 255);
}
