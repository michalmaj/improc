// tests/ml/augment/test_color_extras.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include <opencv2/core.hpp>
#include "improc/ml/augment/color.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/ml/augment/compose.hpp"

using namespace improc::core;
using namespace improc::ml;

// ---- RandomGrayscale ----

TEST(ColorExtrasAugTest, RandomGrayscalePreservesSizeAndType) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomGrayscale{}.p(1.0f)(img, rng);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(ColorExtrasAugTest, RandomGrayscaleP1MakesGray) {
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomGrayscale{}.p(1.0f)(img, rng);
    // All channels equal (grayscale BGR)
    auto px = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_EQ(px[0], px[1]);
    EXPECT_EQ(px[1], px[2]);
}

TEST(ColorExtrasAugTest, RandomGrayscaleP0Identity) {
    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomGrayscale{}.p(0.0f)(img, rng);
    cv::Mat diff;
    cv::absdiff(img.mat(), result.mat(), diff);
    EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(ColorExtrasAugTest, RandomGrayscaleGrayInputUnchanged) {
    cv::Mat mat(8, 8, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(mat);
    std::mt19937 rng(42);
    Image<Gray> result = RandomGrayscale{}.p(1.0f)(img, rng);
    EXPECT_EQ(result.rows(), 8);
    EXPECT_EQ(result.cols(), 8);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(ColorExtrasAugTest, RandomGrayscaleInvalidPThrows) {
    EXPECT_THROW(RandomGrayscale{}.p(-0.1f), improc::ParameterError);
    EXPECT_THROW(RandomGrayscale{}.p(1.1f),  improc::ParameterError);
}

TEST(ColorExtrasAugTest, RandomGrayscaleBindRngPipelineOp) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomGrayscale{}.p(0.0f).bind(rng);
    EXPECT_EQ(result.rows(), 32);
}

// ---- RandomSolarize ----

TEST(ColorExtrasAugTest, RandomSolarizePreservesSizeAndType) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomSolarize{}.p(1.0f)(img, rng);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(ColorExtrasAugTest, RandomSolarizeP0Identity) {
    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomSolarize{}.p(0.0f)(img, rng);
    cv::Mat diff;
    cv::absdiff(img.mat(), result.mat(), diff);
    EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(ColorExtrasAugTest, RandomSolarizeInvertsAboveThreshold) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    // threshold=128: pixel 200 >= 128 → 255-200 = 55
    Image<BGR> result = RandomSolarize{}.threshold(128).p(1.0f)(img, rng);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 55);
}

TEST(ColorExtrasAugTest, RandomSolarizeInvalidThresholdThrows) {
    EXPECT_THROW(RandomSolarize{}.threshold(-1),  improc::ParameterError);
    EXPECT_THROW(RandomSolarize{}.threshold(256), improc::ParameterError);
}

TEST(ColorExtrasAugTest, RandomSolarizeInvalidPThrows) {
    EXPECT_THROW(RandomSolarize{}.p(-0.1f), improc::ParameterError);
    EXPECT_THROW(RandomSolarize{}.p(1.1f),  improc::ParameterError);
}

// ---- RandomPosterize ----

TEST(ColorExtrasAugTest, RandomPosterizePreservesSizeAndType) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomPosterize{}.p(1.0f)(img, rng);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(ColorExtrasAugTest, RandomPosterizeReducesBits) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(255, 255, 255));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    // 1 bit: mask = 0x80 → pixel 255 & 0x80 = 128
    Image<BGR> result = RandomPosterize{}.bits(1).p(1.0f)(img, rng);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 128);
}

TEST(ColorExtrasAugTest, RandomPosterizeInvalidBitsThrows) {
    EXPECT_THROW(RandomPosterize{}.bits(0), improc::ParameterError);
    EXPECT_THROW(RandomPosterize{}.bits(9), improc::ParameterError);
}

TEST(ColorExtrasAugTest, RandomPosterizeInvalidPThrows) {
    EXPECT_THROW(RandomPosterize{}.p(-0.1f), improc::ParameterError);
    EXPECT_THROW(RandomPosterize{}.p(1.1f),  improc::ParameterError);
}

// ---- RandomEqualize ----

TEST(ColorExtrasAugTest, RandomEqualizePreservesSizeAndType) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomEqualize{}.p(1.0f)(img, rng);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(ColorExtrasAugTest, RandomEqualizeGrayPreservesSizeAndType) {
    cv::Mat mat(32, 32, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(mat);
    std::mt19937 rng(42);
    Image<Gray> result = RandomEqualize{}.p(1.0f)(img, rng);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(ColorExtrasAugTest, RandomEqualizeP0Identity) {
    cv::Mat mat(16, 16, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomEqualize{}.p(0.0f)(img, rng);
    cv::Mat diff;
    cv::absdiff(img.mat(), result.mat(), diff);
    EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(ColorExtrasAugTest, RandomEqualizeInvalidPThrows) {
    EXPECT_THROW(RandomEqualize{}.p(-0.1f), improc::ParameterError);
    EXPECT_THROW(RandomEqualize{}.p(1.1f),  improc::ParameterError);
}

TEST(ColorExtrasAugTest, RandomEqualizeBindRngPipelineOp) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomEqualize{}.p(0.0f).bind(rng);
    EXPECT_EQ(result.rows(), 32);
}

// ---- Compose integration ----

TEST(ColorExtrasAugTest, ComposeColorExtras) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto pipeline = Compose<BGR>{}
        .add(RandomGrayscale{}.p(0.5f))
        .add(RandomSolarize{}.threshold(128).p(0.5f))
        .add(RandomPosterize{}.bits(4).p(0.5f))
        .add(RandomEqualize{}.p(0.5f));
    Image<BGR> result = pipeline(img, rng);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}
