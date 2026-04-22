// tests/core/ops/test_hsv_conversion.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(HsvConversionTest, ConvertBGRtoHSVOutputType) {
    Image<BGR> src(cv::Mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 255)));
    Image<HSV> result = convert<HSV, BGR>(src);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 4);
}

TEST(HsvConversionTest, ConvertHSVtoBGROutputType) {
    cv::Mat hsv_mat(4, 4, CV_8UC3, cv::Scalar(0, 255, 255));
    Image<HSV> src(hsv_mat);
    Image<BGR> result = convert<BGR, HSV>(src);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(HsvConversionTest, RoundTripPreservesPixels) {
    cv::Mat bgr_mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 255));  // pure red
    Image<BGR> src(bgr_mat);
    Image<BGR> result = convert<BGR, HSV>(convert<HSV, BGR>(src));
    double diff = cv::norm(src.mat(), result.mat(), cv::NORM_INF);
    EXPECT_LE(diff, 2.0);
}

TEST(HsvConversionTest, KnownPixelRedBGRtoHSV) {
    cv::Mat bgr_mat(1, 1, CV_8UC3, cv::Scalar(0, 0, 255));
    Image<BGR> src(bgr_mat);
    Image<HSV> result = convert<HSV, BGR>(src);
    cv::Vec3b px = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_LE(px[0], 1);
    EXPECT_EQ(px[1], 255);
    EXPECT_EQ(px[2], 255);
}

TEST(HsvConversionTest, ToHSVOpMatchesFreeFunction) {
    Image<BGR> src(cv::Mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<HSV> via_op       = src | ToHSV{};
    Image<HSV> via_function = convert<HSV, BGR>(src);
    double diff = cv::norm(via_op.mat(), via_function.mat(), cv::NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

TEST(HsvConversionTest, ToBGRFromHSVOpMatchesFreeFunction) {
    cv::Mat hsv_mat(4, 4, CV_8UC3, cv::Scalar(30, 200, 180));
    Image<HSV> src(hsv_mat);
    Image<BGR> via_op       = src | ToBGR{};
    Image<BGR> via_function = convert<BGR, HSV>(src);
    double diff = cv::norm(via_op.mat(), via_function.mat(), cv::NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

TEST(HsvConversionTest, ToBGRFromGrayStillWorks) {
    Image<Gray> gray(cv::Mat(4, 4, CV_8UC1, cv::Scalar(128)));
    Image<BGR>  result = gray | ToBGR{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(HSVConversionTest, KnownPixelGreenBGRtoHSV) {
    cv::Mat mat(1, 1, CV_8UC3, cv::Scalar(0, 255, 0)); // pure green in BGR
    Image<BGR> bgr(mat);
    Image<HSV> hsv = bgr | ToHSV{};
    cv::Vec3b px = hsv.mat().at<cv::Vec3b>(0, 0);
    EXPECT_EQ(px[0], 60);  // H=60
    EXPECT_EQ(px[1], 255); // S=255
    EXPECT_EQ(px[2], 255); // V=255
}
