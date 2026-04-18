// tests/core/ops/test_bilateral_filter.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/core/ops/bilateral_filter.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(BilateralFilterTest, ZeroDiameterThrows) {
    EXPECT_THROW(BilateralFilter{}.diameter(0), improc::ParameterError);
}
TEST(BilateralFilterTest, NegativeDiameterThrows) {
    EXPECT_THROW(BilateralFilter{}.diameter(-1), improc::ParameterError);
}
TEST(BilateralFilterTest, ZeroSigmaColorThrows) {
    EXPECT_THROW(BilateralFilter{}.sigma_color(0.0), improc::ParameterError);
}
TEST(BilateralFilterTest, ZeroSigmaSpaceThrows) {
    EXPECT_THROW(BilateralFilter{}.sigma_space(0.0), improc::ParameterError);
}

TEST(BilateralFilterTest, GrayPreservesSizeAndType) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(128)));
    auto result = BilateralFilter{}(img);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(BilateralFilterTest, BGRPreservesSizeAndType) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = BilateralFilter{}(img);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(BilateralFilterTest, UniformImageUnchanged) {
    // Bilateral filter on a flat image should leave it unchanged
    cv::Mat mat(32, 32, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    auto result = BilateralFilter{}(img);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 100);
    EXPECT_EQ(result.mat().at<uchar>(16, 16), 100);
}

TEST(BilateralFilterTest, PipelineFormGray) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(128)));
    auto result = img | BilateralFilter{}.diameter(5).sigma_color(50).sigma_space(50);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(BilateralFilterTest, PipelineFormBGR) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = img | BilateralFilter{}.diameter(5);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}
