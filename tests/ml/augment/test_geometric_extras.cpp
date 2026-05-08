// tests/ml/augment/test_geometric_extras.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include <opencv2/core.hpp>
#include "improc/ml/augment/geometric.hpp"
#include "improc/ml/augment/compose.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;
using namespace improc::ml;

// ---- RandomZoom ----

TEST(GeometricExtrasAugTest, RandomZoomPreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomZoom{}(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(GeometricExtrasAugTest, RandomZoomGrayPreservesSizeAndType) {
    cv::Mat mat(32, 32, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(mat);
    std::mt19937 rng(42);
    Image<Gray> result = RandomZoom{}(img, rng);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(GeometricExtrasAugTest, RandomZoomInvalidRangeThrows) {
    EXPECT_THROW(RandomZoom{}.range(0.0f, 1.0f), improc::ParameterError);  // min <= 0
    EXPECT_THROW(RandomZoom{}.range(-0.5f, 0.5f), improc::ParameterError); // min < 0
    EXPECT_THROW(RandomZoom{}.range(0.5f, 1.5f), improc::ParameterError);  // max > 1
    EXPECT_THROW(RandomZoom{}.range(0.8f, 0.5f), improc::ParameterError);  // min > max
}

TEST(GeometricExtrasAugTest, RandomZoomBindRngPipelineOp) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomZoom{}.range(0.8f, 1.0f).bind(rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

TEST(GeometricExtrasAugTest, RandomZoomFullScaleIdentity) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    // scale=1.0 → crop full image and resize back → same pixels
    Image<BGR> result = RandomZoom{}.range(1.0f, 1.0f)(img, rng);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    cv::Mat diff;
    cv::absdiff(img.mat(), result.mat(), diff);
    EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);
}

// ---- RandomShear ----

TEST(GeometricExtrasAugTest, RandomShearPreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomShear{}(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(GeometricExtrasAugTest, RandomShearZeroAngleIdentity) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(80, 80, 80));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomShear{}.range(0.0f, 0.0f)(img, rng);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}

TEST(GeometricExtrasAugTest, RandomShearInvalidRangeThrows) {
    EXPECT_THROW(RandomShear{}.range(15.0f, -15.0f), improc::ParameterError);
}

TEST(GeometricExtrasAugTest, RandomShearBindRngPipelineOp) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomShear{}.range(0.0f, 0.0f).bind(rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

// ---- RandomPerspective ----

TEST(GeometricExtrasAugTest, RandomPerspectivePreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomPerspective{}(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(GeometricExtrasAugTest, RandomPerspectiveInvalidDistortionScaleThrows) {
    EXPECT_THROW(RandomPerspective{}.distortion_scale(-0.1f), improc::ParameterError);
    EXPECT_THROW(RandomPerspective{}.distortion_scale(1.1f),  improc::ParameterError);
}

TEST(GeometricExtrasAugTest, RandomPerspectiveZeroDistortionIdentity) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomPerspective{}.distortion_scale(0.0f)(img, rng);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}

TEST(GeometricExtrasAugTest, RandomPerspectiveBindRngPipelineOp) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomPerspective{}.distortion_scale(0.3f).bind(rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

// ---- Compose integration ----

TEST(GeometricExtrasAugTest, ComposeGeometricExtras) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto pipeline = Compose<BGR>{}
        .add(RandomZoom{}.range(0.8f, 1.0f))
        .add(RandomShear{}.range(-10.0f, 10.0f))
        .add(RandomPerspective{}.distortion_scale(0.3f));
    Image<BGR> result = pipeline(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}
