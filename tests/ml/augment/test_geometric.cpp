// tests/ml/augment/test_geometric.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/ml/augment/geometric.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;
using namespace improc::ml;

// ---- RandomFlip ----

TEST(GeometricAugTest, RandomFlipPreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomFlip{}.p(0.5f)(img, rng);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(GeometricAugTest, RandomFlipAlwaysFlipsAtP1) {
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    mat.at<cv::Vec3b>(0, 0) = {255, 0, 0};  // marker at top-left
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomFlip{}.p(1.0f)(img, rng);
    // horizontal flip: top-left marker moves to top-right (col 3)
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 3)[0], 255);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 0);
}

TEST(GeometricAugTest, RandomFlipNeverFlipsAtP0) {
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    mat.at<cv::Vec3b>(0, 0) = {255, 0, 0};
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomFlip{}.p(0.0f)(img, rng);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 255);
}

TEST(GeometricAugTest, RandomFlipInvalidPThrows) {
    EXPECT_THROW(RandomFlip{}.p(1.5f),  std::invalid_argument);
    EXPECT_THROW(RandomFlip{}.p(-0.1f), std::invalid_argument);
}

TEST(GeometricAugTest, RandomFlipBindRngPipelineOp) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    std::mt19937 rng(42);
    Image<Gray> result = img | RandomFlip{}.p(0.0f).bind(rng);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
}

// ---- RandomRotate ----

TEST(GeometricAugTest, RandomRotatePreservesSizeAndType) {
    cv::Mat mat(20, 20, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomRotate{}.range(-15.0f, 15.0f)(img, rng);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(GeometricAugTest, RandomRotateInvalidRangeThrows) {
    EXPECT_THROW(RandomRotate{}.range(10.0f, 5.0f), std::invalid_argument);
}

TEST(GeometricAugTest, RandomRotateInvalidScaleThrows) {
    EXPECT_THROW(RandomRotate{}.scale(0.0f),  std::invalid_argument);
    EXPECT_THROW(RandomRotate{}.scale(-1.0f), std::invalid_argument);
}

TEST(GeometricAugTest, RandomRotateBindRngPipelineOp) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomRotate{}.range(0.0f, 0.0f).bind(rng);
    EXPECT_EQ(result.rows(), 10);
}

// ---- RandomCrop ----

TEST(GeometricAugTest, RandomCropReturnsCorrectSize) {
    cv::Mat mat(20, 20, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomCrop{}.width(10).height(8)(img, rng);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.rows(), 8);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(GeometricAugTest, RandomCropLargerThanImageThrows) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    EXPECT_THROW(RandomCrop{}.width(20).height(5)(img, rng),  std::invalid_argument);
    EXPECT_THROW(RandomCrop{}.width(5).height(20)(img, rng),  std::invalid_argument);
}

TEST(GeometricAugTest, RandomCropMissingDimensionsThrows) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    EXPECT_THROW(RandomCrop{}.width(5)(img, rng),  std::invalid_argument);
    EXPECT_THROW(RandomCrop{}(img, rng),            std::invalid_argument);
}

TEST(GeometricAugTest, RandomCropInvalidDimensionThrows) {
    EXPECT_THROW(RandomCrop{}.width(0),   std::invalid_argument);
    EXPECT_THROW(RandomCrop{}.height(-1), std::invalid_argument);
}

TEST(GeometricAugTest, RandomCropBindRngPipelineOp) {
    cv::Mat mat(20, 20, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomCrop{}.width(10).height(10).bind(rng);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
}

// ---- RandomResize ----

TEST(GeometricAugTest, RandomResizeShorterSideInRange) {
    cv::Mat mat(100, 200, CV_8UC3, cv::Scalar(100, 100, 100));  // landscape: shorter side = 100
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomResize{}.range(50, 80)(img, rng);
    int shorter = std::min(result.rows(), result.cols());
    EXPECT_GE(shorter, 50);
    EXPECT_LE(shorter, 80);
}

TEST(GeometricAugTest, RandomResizeShorterSideInRangePortrait) {
    cv::Mat mat(200, 100, CV_8UC3, cv::Scalar(100, 100, 100));  // portrait: shorter side = cols=100
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomResize{}.range(50, 80)(img, rng);
    int shorter = std::min(result.rows(), result.cols());
    EXPECT_GE(shorter, 50);
    EXPECT_LE(shorter, 80);
}

TEST(GeometricAugTest, RandomResizeInvalidRangeThrows) {
    EXPECT_THROW(RandomResize{}.range(100, 50),  std::invalid_argument);
    EXPECT_THROW(RandomResize{}.range(0, 100),   std::invalid_argument);
}

TEST(GeometricAugTest, RandomResizeBindRngPipelineOp) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomResize{}.range(50, 50).bind(rng);
    EXPECT_EQ(result.rows(), 50);
    EXPECT_EQ(result.cols(), 50);
}
