// tests/core/test_convert.cpp
#include <gtest/gtest.h>
#include "improc/core/convert.hpp"

using namespace improc::core;

TEST(ConvertTest, BGRToGrayPreservesDimensions) {
    cv::Mat mat(50, 60, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> bgr(mat);
    Image<Gray> gray = convert<Gray, BGR>(bgr);
    EXPECT_EQ(gray.mat().type(), CV_8UC1);
    EXPECT_EQ(gray.rows(), 50);
    EXPECT_EQ(gray.cols(), 60);
}

TEST(ConvertTest, GrayToBGRPreservesDimensions) {
    cv::Mat mat(50, 60, CV_8UC1, cv::Scalar(128));
    Image<Gray> gray(mat);
    Image<BGR> bgr = convert<BGR, Gray>(gray);
    EXPECT_EQ(bgr.mat().type(), CV_8UC3);
    EXPECT_EQ(bgr.rows(), 50);
    EXPECT_EQ(bgr.cols(), 60);
}

TEST(ConvertTest, BGRToBGRAAddsAlphaChannel) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(10, 20, 30));
    Image<BGR> bgr(mat);
    Image<BGRA> bgra = convert<BGRA, BGR>(bgr);
    EXPECT_EQ(bgra.mat().type(), CV_8UC4);
}

TEST(ConvertTest, BGRAToBGRDropsAlphaChannel) {
    cv::Mat mat(10, 10, CV_8UC4, cv::Scalar(10, 20, 30, 255));
    Image<BGRA> bgra(mat);
    Image<BGR> bgr = convert<BGR, BGRA>(bgra);
    EXPECT_EQ(bgr.mat().type(), CV_8UC3);
}

TEST(ConvertTest, GrayToFloat32NormalizesValues) {
    cv::Mat mat(1, 1, CV_8UC1, cv::Scalar(255));
    Image<Gray> gray(mat);
    Image<Float32> f = convert<Float32, Gray>(gray);
    EXPECT_EQ(f.mat().type(), CV_32FC1);
    EXPECT_NEAR(f.mat().at<float>(0, 0), 1.0f, 1e-5f);
}

TEST(ConvertTest, GrayToFloat32ZeroIsZero) {
    cv::Mat mat(1, 1, CV_8UC1, cv::Scalar(0));
    Image<Gray> gray(mat);
    Image<Float32> f = convert<Float32, Gray>(gray);
    EXPECT_NEAR(f.mat().at<float>(0, 0), 0.0f, 1e-5f);
}

TEST(ConvertTest, Float32ToGrayScalesValues) {
    cv::Mat mat(1, 1, CV_32FC1, cv::Scalar(1.0f));
    Image<Float32> f(mat);
    Image<Gray> gray = convert<Gray, Float32>(f);
    EXPECT_EQ(gray.mat().type(), CV_8UC1);
    EXPECT_EQ(gray.mat().at<uchar>(0, 0), 255);
}

TEST(ConvertTest, Float32ToGrayZeroIsZero) {
    cv::Mat mat(1, 1, CV_32FC1, cv::Scalar(0.0f));
    Image<Float32> f(mat);
    Image<Gray> gray = convert<Gray, Float32>(f);
    EXPECT_EQ(gray.mat().at<uchar>(0, 0), 0);
}

TEST(ConvertTest, Float32ToGrayPreservesDimensions) {
    cv::Mat mat(50, 60, CV_32FC1, cv::Scalar(0.5f));
    Image<Float32> f(mat);
    Image<Gray> gray = convert<Gray, Float32>(f);
    EXPECT_EQ(gray.rows(), 50);
    EXPECT_EQ(gray.cols(), 60);
}

TEST(ConvertTest, Float32C3ToBGRScalesValues) {
    cv::Mat mat(1, 1, CV_32FC3, cv::Scalar(1.0f, 1.0f, 1.0f));
    Image<Float32C3> f(mat);
    Image<BGR> bgr = convert<BGR, Float32C3>(f);
    EXPECT_EQ(bgr.mat().type(), CV_8UC3);
    auto px = bgr.mat().at<cv::Vec3b>(0, 0);
    EXPECT_EQ(px[0], 255);
    EXPECT_EQ(px[1], 255);
    EXPECT_EQ(px[2], 255);
}

TEST(ConvertTest, Float32C3ToBGRZeroIsZero) {
    cv::Mat mat(1, 1, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
    Image<Float32C3> f(mat);
    Image<BGR> bgr = convert<BGR, Float32C3>(f);
    auto px = bgr.mat().at<cv::Vec3b>(0, 0);
    EXPECT_EQ(px[0], 0);
    EXPECT_EQ(px[1], 0);
    EXPECT_EQ(px[2], 0);
}

TEST(ConvertTest, Float32C3ToBGRPreservesDimensions) {
    cv::Mat mat(50, 60, CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f));
    Image<Float32C3> f(mat);
    Image<BGR> bgr = convert<BGR, Float32C3>(f);
    EXPECT_EQ(bgr.rows(), 50);
    EXPECT_EQ(bgr.cols(), 60);
}
