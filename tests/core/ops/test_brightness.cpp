// tests/core/ops/test_brightness.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(BrightnessTest, PositiveDeltaIncreasesMeanValue) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    double original_mean = cv::mean(img.mat())[0];
    Image<BGR> result = img | Brightness{}.delta(50.0);
    double result_mean = cv::mean(result.mat())[0];
    EXPECT_GT(result_mean, original_mean);
}

TEST(BrightnessTest, NegativeDeltaDecreasesMeanValue) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    double original_mean = cv::mean(img.mat())[0];
    Image<BGR> result = img | Brightness{}.delta(-30.0);
    double result_mean = cv::mean(result.mat())[0];
    EXPECT_LT(result_mean, original_mean);
}

TEST(BrightnessTest, ZeroDeltaReturnsEqualImage) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    double original_mean = cv::mean(img.mat())[0];
    Image<BGR> result = img | Brightness{}.delta(0.0);
    double result_mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(result_mean, original_mean, 1e-5);
}

TEST(BrightnessTest, DefaultDeltaIsZero) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    double original_mean = cv::mean(img.mat())[0];
    Image<BGR> result = Brightness{}(img);
    double result_mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(result_mean, original_mean, 1e-5);
}

TEST(BrightnessTest, ClippingMaxPixelStaysBelow256) {
    cv::Mat mat(1, 1, CV_8UC3, cv::Scalar(250, 250, 250));
    Image<BGR> img(mat);
    Image<BGR> result = img | Brightness{}.delta(100.0);
    cv::Vec3b pixel = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_EQ(pixel[0], 255);
    EXPECT_EQ(pixel[1], 255);
    EXPECT_EQ(pixel[2], 255);
}

TEST(BrightnessTest, WorksOnGray) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = img | Brightness{}.delta(20.0);
    double result_mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(result_mean, 120.0, 1.0);
}

TEST(BrightnessTest, WorksOnBGR) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(50, 50, 50));
    Image<BGR> img(mat);
    Image<BGR> result = img | Brightness{}.delta(10.0);
    double result_mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(result_mean, 60.0, 1.0);
}

TEST(BrightnessTest, WorksOnFloat32C3) {
    cv::Mat mat(5, 5, CV_32FC3, cv::Scalar(0.5f, 0.5f, 0.5f));
    Image<Float32C3> img(mat);
    Image<Float32C3> result = img | Brightness{}.delta(0.1);
    double result_mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(result_mean, 0.6, 0.01);
}
