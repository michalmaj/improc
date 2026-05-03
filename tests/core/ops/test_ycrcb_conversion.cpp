// tests/core/ops/test_ycrcb_conversion.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(YCrCbConversionTest, ConvertBGRtoYCrCbOutputType) {
    Image<BGR> src(cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<YCrCb> result = src | ToYCrCb{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 4);
}

TEST(YCrCbConversionTest, ConvertYCrCbtoBGROutputType) {
    cv::Mat ycrcb_mat(4, 4, CV_8UC3, cv::Scalar(128, 128, 128));
    Image<YCrCb> src(ycrcb_mat);
    Image<BGR> result = src | ToBGR{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 4);
}

TEST(YCrCbConversionTest, RoundTripPreservesPixels) {
    Image<BGR> src(cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<BGR> result = src | ToYCrCb{} | ToBGR{};
    double diff = cv::norm(src.mat(), result.mat(), cv::NORM_INF);
    EXPECT_LE(diff, 2.0);
}

TEST(YCrCbConversionTest, WhiteBGRRoundTrip) {
    Image<BGR> src(cv::Mat(1, 1, CV_8UC3, cv::Scalar(255, 255, 255)));
    Image<BGR> result = src | ToYCrCb{} | ToBGR{};
    cv::Vec3b px = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_NEAR(px[0], 255, 2);
    EXPECT_NEAR(px[1], 255, 2);
    EXPECT_NEAR(px[2], 255, 2);
}

TEST(YCrCbConversionTest, BlackBGRRoundTrip) {
    Image<BGR> src(cv::Mat(1, 1, CV_8UC3, cv::Scalar(0, 0, 0)));
    Image<BGR> result = src | ToYCrCb{} | ToBGR{};
    cv::Vec3b px = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_NEAR(px[0], 0, 2);
    EXPECT_NEAR(px[1], 0, 2);
    EXPECT_NEAR(px[2], 0, 2);
}

TEST(YCrCbConversionTest, ToYCrCbOpMatchesFreeFunction) {
    Image<BGR> src(cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<YCrCb> via_op       = src | ToYCrCb{};
    Image<YCrCb> via_function = convert<YCrCb, BGR>(src);
    double diff = cv::norm(via_op.mat(), via_function.mat(), cv::NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

TEST(YCrCbConversionTest, ToBGRFromYCrCbOpMatchesFreeFunction) {
    Image<BGR> bgr(cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<YCrCb> ycrcb = convert<YCrCb, BGR>(bgr);
    Image<BGR> via_op       = ycrcb | ToBGR{};
    Image<BGR> via_function = convert<BGR, YCrCb>(ycrcb);
    double diff = cv::norm(via_op.mat(), via_function.mat(), cv::NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

TEST(YCrCbConversionTest, ToBGRFromLABStillWorks) {
    Image<BGR> bgr(cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<LAB> lab = convert<LAB, BGR>(bgr);
    Image<BGR> result = lab | ToBGR{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(YCrCbConversionTest, PipelineSyntax) {
    Image<BGR> src(cv::Mat(4, 4, CV_8UC3, cv::Scalar(50, 100, 150)));
    auto result = src | ToYCrCb{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}
