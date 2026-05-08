// tests/ml/augment/test_blur.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include <opencv2/core.hpp>
#include "improc/ml/augment/blur.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/ml/augment/compose.hpp"

using namespace improc::core;
using namespace improc::ml;

// ---- RandomBlur ----

TEST(BlurAugTest, RandomBlurPreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomBlur{}(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(BlurAugTest, RandomBlurGrayPreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(mat);
    std::mt19937 rng(42);
    // Default types include Bilateral — Gray is 8-bit so no throw
    Image<Gray> result = RandomBlur{}(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(BlurAugTest, RandomBlurGaussianOnly) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomBlur{}.types({RandomBlur::Type::Gaussian})(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

TEST(BlurAugTest, RandomBlurMedianOnly) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomBlur{}.types({RandomBlur::Type::Median})(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

TEST(BlurAugTest, RandomBlurBilateralOnFloat32Throws) {
    cv::Mat mat(64, 64, CV_32FC1, cv::Scalar(0.5f));
    Image<Float32> img(mat);
    std::mt19937 rng(42);
    // Default types include Bilateral — Float32 throws at call time
    EXPECT_THROW(RandomBlur{}(img, rng), improc::ParameterError);
}

TEST(BlurAugTest, RandomBlurBilateralExcludedFromFloat32Works) {
    cv::Mat mat(64, 64, CV_32FC1, cv::Scalar(0.5f));
    Image<Float32> img(mat);
    std::mt19937 rng(42);
    Image<Float32> result = RandomBlur{}
        .types({RandomBlur::Type::Gaussian, RandomBlur::Type::Median})(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

TEST(BlurAugTest, RandomBlurInvalidEmptyTypesThrows) {
    EXPECT_THROW(RandomBlur{}.types({}), improc::ParameterError);
}

TEST(BlurAugTest, RandomBlurInvalidKernelSizeThrows) {
    EXPECT_THROW(RandomBlur{}.kernel_size(2, 7),  improc::ParameterError); // min < 3
    EXPECT_THROW(RandomBlur{}.kernel_size(3, 33), improc::ParameterError); // max > 31
    EXPECT_THROW(RandomBlur{}.kernel_size(4, 7),  improc::ParameterError); // min even
    EXPECT_THROW(RandomBlur{}.kernel_size(3, 8),  improc::ParameterError); // max even
    EXPECT_THROW(RandomBlur{}.kernel_size(9, 7),  improc::ParameterError); // min > max
}

TEST(BlurAugTest, RandomBlurBindRngPipelineOp) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomBlur{}.types({RandomBlur::Type::Gaussian}).bind(rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

// ---- RandomSharpness ----

TEST(BlurAugTest, RandomSharpnessPreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomSharpness{}.p(1.0f)(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(BlurAugTest, RandomSharpnessP0Identity) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomSharpness{}.p(0.0f)(img, rng);
    cv::Mat diff;
    cv::absdiff(img.mat(), result.mat(), diff);
    EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(BlurAugTest, RandomSharpnessZeroStrengthIdentity) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    // strength=0 → img + 0*(img-blurred) = img (with rounding may differ by 1)
    Image<BGR> result = RandomSharpness{}.range(0.0f, 0.0f).p(1.0f)(img, rng);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}

TEST(BlurAugTest, RandomSharpnessInvalidRangeThrows) {
    EXPECT_THROW(RandomSharpness{}.range(-0.1f, 1.0f), improc::ParameterError); // min < 0
    EXPECT_THROW(RandomSharpness{}.range(1.0f, 0.5f),  improc::ParameterError); // min > max
}

TEST(BlurAugTest, RandomSharpnessInvalidPThrows) {
    EXPECT_THROW(RandomSharpness{}.p(-0.1f), improc::ParameterError);
    EXPECT_THROW(RandomSharpness{}.p(1.1f),  improc::ParameterError);
}

TEST(BlurAugTest, RandomSharpnessBindRngPipelineOp) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomSharpness{}.p(0.0f).bind(rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

TEST(BlurAugTest, RandomSharpnessFloat32ClampsToOne) {
    cv::Mat mat(32, 32, CV_32FC1, cv::Scalar(0.9f));
    Image<Float32> img(mat);
    std::mt19937 rng(42);
    Image<Float32> result = RandomSharpness{}.range(5.0f, 5.0f).p(1.0f)(img, rng);
    double minVal, maxVal;
    cv::minMaxLoc(result.mat(), &minVal, &maxVal);
    EXPECT_LE(maxVal, 1.0);
    EXPECT_GE(minVal, 0.0);
}

// ---- Compose integration ----

TEST(BlurAugTest, ComposeBlurOps) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto pipeline = Compose<BGR>{}
        .add(RandomBlur{}.types({RandomBlur::Type::Gaussian, RandomBlur::Type::Median})
                         .kernel_size(3, 7))
        .add(RandomSharpness{}.range(0.0f, 1.0f).p(0.5f));
    Image<BGR> result = pipeline(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}
