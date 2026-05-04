// tests/core/ops/test_pyramid.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(PyrDownTest, HalvesSizeGray) {
    Image<Gray> src(cv::Mat(100, 100, CV_8UC1, cv::Scalar(128)));
    Image<Gray> dst = src | PyrDown{};
    EXPECT_EQ(dst.rows(), 50);
    EXPECT_EQ(dst.cols(), 50);
}

TEST(PyrDownTest, HalvesSizeBGR) {
    Image<BGR> src(cv::Mat(100, 100, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<BGR> dst = src | PyrDown{};
    EXPECT_EQ(dst.rows(), 50);
    EXPECT_EQ(dst.cols(), 50);
}

TEST(PyrDownTest, PreservesTypeGray) {
    Image<Gray> src(cv::Mat(64, 64, CV_8UC1, cv::Scalar(128)));
    Image<Gray> dst = src | PyrDown{};
    EXPECT_EQ(dst.mat().type(), CV_8UC1);
}

TEST(PyrDownTest, PreservesTypeBGR) {
    Image<BGR> src(cv::Mat(64, 64, CV_8UC3, cv::Scalar(0, 128, 255)));
    Image<BGR> dst = src | PyrDown{};
    EXPECT_EQ(dst.mat().type(), CV_8UC3);
}

TEST(PyrDownTest, PipelineSyntax) {
    Image<Gray> src(cv::Mat(64, 64, CV_8UC1, cv::Scalar(100)));
    auto result = src | PyrDown{};
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}

TEST(PyrUpTest, DoublesSizeGray) {
    Image<Gray> src(cv::Mat(50, 50, CV_8UC1, cv::Scalar(128)));
    Image<Gray> dst = src | PyrUp{};
    EXPECT_EQ(dst.rows(), 100);
    EXPECT_EQ(dst.cols(), 100);
}

TEST(PyrUpTest, DoublesSizeBGR) {
    Image<BGR> src(cv::Mat(50, 50, CV_8UC3, cv::Scalar(100, 150, 200)));
    Image<BGR> dst = src | PyrUp{};
    EXPECT_EQ(dst.rows(), 100);
    EXPECT_EQ(dst.cols(), 100);
}

TEST(PyrUpTest, PreservesTypeGray) {
    Image<Gray> src(cv::Mat(32, 32, CV_8UC1, cv::Scalar(128)));
    Image<Gray> dst = src | PyrUp{};
    EXPECT_EQ(dst.mat().type(), CV_8UC1);
}

TEST(PyrUpTest, PreservesTypeBGR) {
    Image<BGR> src(cv::Mat(32, 32, CV_8UC3, cv::Scalar(0, 128, 255)));
    Image<BGR> dst = src | PyrUp{};
    EXPECT_EQ(dst.mat().type(), CV_8UC3);
}

TEST(PyrUpTest, PipelineSyntax) {
    Image<Gray> src(cv::Mat(32, 32, CV_8UC1, cv::Scalar(100)));
    auto result = src | PyrUp{};
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

TEST(PyrDownTest, ChainedDownThenUpPreservesType) {
    Image<Gray> src(cv::Mat(100, 100, CV_8UC1, cv::Scalar(128)));
    Image<Gray> result = src | PyrDown{} | PyrUp{};
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 100);
    EXPECT_EQ(result.cols(), 100);
}
