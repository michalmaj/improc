// tests/core/ops/test_gamma.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/core/ops/gamma.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(GammaCorrectionTest, ZeroGammaThrows) {
    EXPECT_THROW(GammaCorrection{}.gamma(0.0f), improc::ParameterError);
}

TEST(GammaCorrectionTest, NegativeGammaThrows) {
    EXPECT_THROW(GammaCorrection{}.gamma(-1.0f), improc::ParameterError);
}

TEST(GammaCorrectionTest, GrayPreservesSizeAndType) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(128)));
    auto result = GammaCorrection{}.gamma(1.0f)(img);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(GammaCorrectionTest, BGRPreservesSizeAndType) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(100, 120, 140)));
    auto result = GammaCorrection{}.gamma(2.0f)(img);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(GammaCorrectionTest, IdentityGammaLeavesImageUnchanged) {
    cv::Mat mat(4, 4, CV_8UC1, cv::Scalar(200));
    Image<Gray> img(mat);
    auto result = GammaCorrection{}.gamma(1.0f)(img);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 200);
}

TEST(GammaCorrectionTest, GammaLessThanOneBrightens) {
    // pixel=128, gamma=0.5 → 255 * (128/255)^0.5 ≈ 181
    cv::Mat mat(4, 4, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(mat);
    auto dark   = GammaCorrection{}.gamma(2.0f)(img);
    auto bright = GammaCorrection{}.gamma(0.5f)(img);
    EXPECT_LT(dark.mat().at<uchar>(0, 0), mat.at<uchar>(0, 0));
    EXPECT_GT(bright.mat().at<uchar>(0, 0), mat.at<uchar>(0, 0));
}

TEST(GammaCorrectionTest, PipelineForm) {
    Image<BGR> img(cv::Mat(16, 16, CV_8UC3, cv::Scalar(100, 100, 100)));
    auto result = img | GammaCorrection{}.gamma(1.5f);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(GammaCorrectionTest, Float32ClampedToUnitRange) {
    cv::Mat mat(4, 4, CV_32FC1, cv::Scalar(0.5f));
    Image<Float32> img(mat);
    auto result = GammaCorrection{}.gamma(2.0f)(img);
    double min_val, max_val;
    cv::minMaxLoc(result.mat(), &min_val, &max_val);
    EXPECT_GE(min_val, 0.0);
    EXPECT_LE(max_val, 1.0);
}
