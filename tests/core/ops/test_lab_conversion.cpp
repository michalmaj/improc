// tests/core/ops/test_lab_conversion.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(LabConversionTest, ConvertBGRtoLABOutputType) {
    Image<BGR> src(cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<LAB> result = src | ToLAB{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 4);
}

TEST(LabConversionTest, ConvertLABtoBGROutputType) {
    cv::Mat lab_mat(4, 4, CV_8UC3, cv::Scalar(128, 128, 128));
    Image<LAB> src(lab_mat);
    Image<BGR> result = src | ToBGR{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 4);
}

TEST(LabConversionTest, RoundTripPreservesPixels) {
    // BGR → LAB → BGR: 8-bit quantization may introduce ≤ 2 per channel
    Image<BGR> src(cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<BGR> result = src | ToLAB{} | ToBGR{};
    double diff = cv::norm(src.mat(), result.mat(), cv::NORM_INF);
    EXPECT_LE(diff, 2.0);
}

TEST(LabConversionTest, WhiteBGRRoundTrip) {
    // Pure white is the L*a*b* neutral point — should survive round-trip within tolerance
    Image<BGR> src(cv::Mat(1, 1, CV_8UC3, cv::Scalar(255, 255, 255)));
    Image<BGR> result = src | ToLAB{} | ToBGR{};
    cv::Vec3b px = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_NEAR(px[0], 255, 2);
    EXPECT_NEAR(px[1], 255, 2);
    EXPECT_NEAR(px[2], 255, 2);
}

TEST(LabConversionTest, BlackBGRRoundTrip) {
    Image<BGR> src(cv::Mat(1, 1, CV_8UC3, cv::Scalar(0, 0, 0)));
    Image<BGR> result = src | ToLAB{} | ToBGR{};
    cv::Vec3b px = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_NEAR(px[0], 0, 2);
    EXPECT_NEAR(px[1], 0, 2);
    EXPECT_NEAR(px[2], 0, 2);
}

TEST(LabConversionTest, ToLABOpMatchesFreeFunction) {
    Image<BGR> src(cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<LAB> via_op       = src | ToLAB{};
    Image<LAB> via_function = convert<LAB, BGR>(src);
    double diff = cv::norm(via_op.mat(), via_function.mat(), cv::NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

TEST(LabConversionTest, ToBGRFromLABOpMatchesFreeFunction) {
    Image<BGR> bgr(cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<LAB> lab = convert<LAB, BGR>(bgr);
    Image<BGR> via_op       = lab | ToBGR{};
    Image<BGR> via_function = convert<BGR, LAB>(lab);
    double diff = cv::norm(via_op.mat(), via_function.mat(), cv::NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

TEST(LabConversionTest, ToBGRFromHSVStillWorks) {
    // Backward compat: existing HSV→BGR overload still compiles after LAB overload added
    cv::Mat hsv_mat(4, 4, CV_8UC3, cv::Scalar(30, 200, 180));
    Image<HSV> src(hsv_mat);
    Image<BGR> result = src | ToBGR{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(LabConversionTest, PipelineSyntax) {
    Image<BGR> src(cv::Mat(4, 4, CV_8UC3, cv::Scalar(50, 100, 150)));
    auto result = src | ToLAB{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}
