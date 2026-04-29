// tests/core/ops/test_adaptive_threshold.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/core/ops/adaptive_threshold.hpp"

using namespace improc::core;

// Helper: uniform Gray image
static Image<Gray> make_gray(int rows, int cols, uchar value) {
    cv::Mat mat(rows, cols, CV_8UC1, cv::Scalar(value));
    return Image<Gray>{mat};
}

TEST(AdaptiveThresholdTest, ProducesSameSizeOutput) {
    Image<Gray> img = make_gray(20, 20, 200);
    Image<Gray> result = img | AdaptiveThreshold{};
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(AdaptiveThresholdTest, OutputTypeIsGray) {
    Image<Gray> img = make_gray(20, 20, 200);
    Image<Gray> result = img | AdaptiveThreshold{};
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(AdaptiveThresholdTest, GaussianMethodAllPixelsBelowThresholdAreZero) {
    // Uniform bright image, large C → local_mean - C is low → all pixels above → all 255
    Image<Gray> img = make_gray(20, 20, 200);
    Image<Gray> result = img | AdaptiveThreshold{}.block_size(11).C(50);
    EXPECT_EQ(cv::countNonZero(result.mat()), 400);  // all 20*20 pixels = 255
}

TEST(AdaptiveThresholdTest, MeanMethodAllPixelsAboveThresholdAreMaxValue) {
    // Uniform bright image, large C → all pixels above local mean - C
    Image<Gray> img = make_gray(20, 20, 200);
    Image<Gray> result = img
        | AdaptiveThreshold{}.method(AdaptiveMethod::Mean).block_size(11).C(50);
    EXPECT_EQ(cv::countNonZero(result.mat()), 400);
}

TEST(AdaptiveThresholdTest, InvertProducesComplementOnUniformImage) {
    // Bright uniform image with C=50: Binary → all 255, BinaryInv → all 0
    Image<Gray> img = make_gray(20, 20, 200);
    Image<Gray> normal   = img | AdaptiveThreshold{}.block_size(11).C(50);
    Image<Gray> inverted = img | AdaptiveThreshold{}.block_size(11).C(50).invert();
    EXPECT_EQ(cv::countNonZero(normal.mat()),   400);  // all 255
    EXPECT_EQ(cv::countNonZero(inverted.mat()),   0);  // all 0
}

TEST(AdaptiveThresholdTest, CustomMaxValueRespected) {
    // Bright uniform image, C=50 → all pixels above threshold → output = max_value(128)
    Image<Gray> img = make_gray(20, 20, 200);
    Image<Gray> result = img | AdaptiveThreshold{}.block_size(11).C(50).max_value(128);
    // All non-zero pixels must equal 128 (not 255)
    cv::Mat out = result.mat();
    EXPECT_EQ(cv::countNonZero(out == 255), 0);
    EXPECT_EQ(cv::countNonZero(out == 128), 400);
}

TEST(AdaptiveThresholdTest, PipelineSyntaxWorks) {
    Image<Gray> img = make_gray(20, 20, 128);
    Image<Gray> result = img | AdaptiveThreshold{};
    EXPECT_EQ(result.rows(), 20);
}

TEST(AdaptiveThresholdTest, ThrowsOnEvenBlockSize) {
    EXPECT_THROW(AdaptiveThreshold{}.block_size(4), improc::ParameterError);
}

TEST(AdaptiveThresholdTest, ThrowsOnBlockSizeOne) {
    EXPECT_THROW(AdaptiveThreshold{}.block_size(1), improc::ParameterError);
}

TEST(AdaptiveThresholdTest, ThrowsOnBlockSizeTwo) {
    EXPECT_THROW(AdaptiveThreshold{}.block_size(2), improc::ParameterError);
}

TEST(AdaptiveThresholdTest, CustomBlockSizeWorks) {
    // 30x30 image, block_size=15 (odd, >= 3, fits in image)
    cv::Mat mat(30, 30, CV_8UC1, cv::Scalar(200));
    Image<Gray> img{mat};
    EXPECT_NO_THROW((img | AdaptiveThreshold{}.block_size(15).C(50)));
}
