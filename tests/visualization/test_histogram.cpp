// tests/visualization/test_histogram.cpp
#include <gtest/gtest.h>
#include "improc/visualization/histogram.hpp"

using namespace improc::core;
using namespace improc::visualization;

TEST(HistogramTest, BGROutputIsCV8UC3) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(128, 64, 32));
    Image<BGR> img(mat);
    Image<BGR> result = img | Histogram{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(HistogramTest, BGRDefaultSize) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(128, 64, 32));
    Image<BGR> img(mat);
    Image<BGR> result = img | Histogram{};
    EXPECT_EQ(result.cols(), 512);
    EXPECT_EQ(result.rows(), 256);
}

TEST(HistogramTest, GrayOutputIsCV8UC3) {
    cv::Mat mat(100, 100, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(mat);
    Image<BGR> result = img | Histogram{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(HistogramTest, Float32OutputIsCV8UC3) {
    cv::Mat mat(100, 100, CV_32FC1);
    mat.setTo(0.5f);
    mat.at<float>(0, 0) = 0.0f;
    mat.at<float>(0, 1) = 1.0f;
    Image<Float32> img(mat);
    Image<BGR> result = img | Histogram{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(HistogramTest, CustomBinsAndSize) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(200, 100, 50));
    Image<BGR> img(mat);
    Image<BGR> result = img | Histogram{}.bins(128).width(400).height(200);
    EXPECT_EQ(result.cols(), 400);
    EXPECT_EQ(result.rows(), 200);
}

TEST(HistogramTest, ZeroBinsThrows) {
    EXPECT_THROW(Histogram{}.bins(0), std::invalid_argument);
}

TEST(HistogramTest, NegativeBinsThrows) {
    EXPECT_THROW(Histogram{}.bins(-1), std::invalid_argument);
}

TEST(HistogramTest, ZeroWidthThrows) {
    EXPECT_THROW(Histogram{}.width(0), std::invalid_argument);
}

TEST(HistogramTest, NegativeWidthThrows) {
    EXPECT_THROW(Histogram{}.width(-1), std::invalid_argument);
}

TEST(HistogramTest, ZeroHeightThrows) {
    EXPECT_THROW(Histogram{}.height(0), std::invalid_argument);
}

TEST(HistogramTest, NegativeHeightThrows) {
    EXPECT_THROW(Histogram{}.height(-1), std::invalid_argument);
}
