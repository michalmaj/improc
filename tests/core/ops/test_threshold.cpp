// tests/core/ops/test_threshold.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/threshold.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(ThresholdTest, BinaryOnGrayPreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = Threshold{}(img);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(ThresholdTest, BinaryInvOnGrayPreservesSize) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = Threshold{}.mode(ThresholdMode::BinaryInv)(img);
    EXPECT_EQ(result.rows(), 10);
}

TEST(ThresholdTest, TruncateOnGrayPreservesSize) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = Threshold{}.mode(ThresholdMode::Truncate)(img);
    EXPECT_EQ(result.rows(), 10);
}

TEST(ThresholdTest, ToZeroOnGrayPreservesSize) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = Threshold{}.mode(ThresholdMode::ToZero)(img);
    EXPECT_EQ(result.rows(), 10);
}

TEST(ThresholdTest, ToZeroInvOnGrayPreservesSize) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = Threshold{}.mode(ThresholdMode::ToZeroInv)(img);
    EXPECT_EQ(result.rows(), 10);
}

TEST(ThresholdTest, OtsuOnGrayPreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = Threshold{}.mode(ThresholdMode::Otsu)(img);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(ThresholdTest, OtsuOnBGRThrowsRuntimeError) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    EXPECT_THROW(Threshold{}.mode(ThresholdMode::Otsu)(img), std::runtime_error);
}

TEST(ThresholdTest, BinaryProducesCorrectValues) {
    // pixel value 200 > threshold 127 → output = 255 (max_value default)
    cv::Mat mat(1, 1, CV_8UC1, cv::Scalar(200));
    Image<Gray> img(mat);
    Image<Gray> result = Threshold{}.value(127).mode(ThresholdMode::Binary)(img);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 255);
}

TEST(ThresholdTest, PipelineOp) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(200));
    Image<Gray> img(mat);
    Image<Gray> result = img | Threshold{}.value(100);
    EXPECT_EQ(result.rows(), 10);
}
