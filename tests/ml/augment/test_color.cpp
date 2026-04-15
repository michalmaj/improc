// tests/ml/augment/test_color.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/ml/augment/color.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;
using namespace improc::ml;

// ---- RandomBrightness ----

TEST(ColorAugTest, RandomBrightnessPreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomBrightness{}(img, rng);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(ColorAugTest, RandomBrightnessChangesPixels) {
    // factor > 1 brightens; at factor=2 pixel(100) → 200
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    // force factor = 2 by using range(2.0, 2.0)
    Image<BGR> result = RandomBrightness{}.range(2.0f, 2.0f)(img, rng);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 200);
}

TEST(ColorAugTest, RandomBrightnessClampAt255) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomBrightness{}.range(2.0f, 2.0f)(img, rng);
    EXPECT_LE(result.mat().at<cv::Vec3b>(0, 0)[0], 255);
}

TEST(ColorAugTest, RandomBrightnessInvalidRangeThrows) {
    EXPECT_THROW(RandomBrightness{}.range(0.0f, 1.0f), std::invalid_argument);  // low <= 0
    EXPECT_THROW(RandomBrightness{}.range(1.5f, 1.0f), std::invalid_argument);  // low > high
}

TEST(ColorAugTest, RandomBrightnessBindRngPipelineOp) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomBrightness{}.range(1.0f, 1.0f).bind(rng);
    EXPECT_EQ(result.rows(), 10);
}

// ---- RandomContrast ----

TEST(ColorAugTest, RandomContrastPreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomContrast{}(img, rng);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(ColorAugTest, RandomContrastUniformImageUnchanged) {
    // alpha * mean + (1-alpha) * mean = mean regardless of alpha
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomContrast{}.range(0.5f, 2.0f)(img, rng);
    // uniform image: contrast adjustment leaves value = mean = 100
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 100);
}

TEST(ColorAugTest, RandomContrastInvalidRangeThrows) {
    EXPECT_THROW(RandomContrast{}.range(0.0f, 1.0f), std::invalid_argument);  // low <= 0
    EXPECT_THROW(RandomContrast{}.range(1.5f, 1.0f), std::invalid_argument);  // low > high
}

// ---- ColorJitter ----

TEST(ColorAugTest, ColorJitterOnBGRPreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 50));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = ColorJitter{}(img, rng);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(ColorAugTest, ColorJitterChangesPixels) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 150, 50));
    Image<BGR> img(mat);
    std::mt19937 rng(0);
    Image<BGR> r1 = ColorJitter{}.brightness(0.5f, 0.5f)(img, rng);
    rng.seed(0);
    Image<BGR> r2 = ColorJitter{}.brightness(1.0f, 1.0f)(img, rng);
    // r1 (factor=0.5) should have lower blue channel than r2 (factor=1.0)
    EXPECT_LT(r1.mat().at<cv::Vec3b>(0, 0)[0], r2.mat().at<cv::Vec3b>(0, 0)[0]);
}

TEST(ColorAugTest, ColorJitterInvalidBrightnessThrows) {
    EXPECT_THROW(ColorJitter{}.brightness(0.0f, 1.0f), std::invalid_argument);
    EXPECT_THROW(ColorJitter{}.brightness(1.5f, 1.0f), std::invalid_argument);
}

TEST(ColorAugTest, ColorJitterInvalidHueThrows) {
    EXPECT_THROW(ColorJitter{}.hue(-200.0f, 0.0f), std::invalid_argument);
    EXPECT_THROW(ColorJitter{}.hue(10.0f, 5.0f),   std::invalid_argument);
}

TEST(ColorAugTest, ColorJitterBindRngPipelineOp) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | ColorJitter{}.bind(rng);
    EXPECT_EQ(result.rows(), 10);
}

// Concept check: BGRFormat enforced at compile time
static_assert(!improc::core::BGRFormat<improc::core::Gray>,
    "BGRFormat concept must reject Gray");
static_assert(improc::core::BGRFormat<improc::core::BGR>,
    "BGRFormat concept must accept BGR");
