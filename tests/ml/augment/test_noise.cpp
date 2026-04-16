// tests/ml/augment/test_noise.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/ml/augment/noise.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;
using namespace improc::ml;

// ---- RandomGaussianNoise ----

TEST(NoiseAugTest, GaussianNoisePreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomGaussianNoise{}(img, rng);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(NoiseAugTest, GaussianNoiseChangesPixels) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomGaussianNoise{}.std_dev(30.0f, 30.0f)(img, rng);
    // With large std_dev, at least one pixel should differ from 100
    bool changed = false;
    for (int r = 0; r < 10 && !changed; ++r)
        for (int c = 0; c < 10 && !changed; ++c)
            if (result.mat().at<cv::Vec3b>(r, c)[0] != 100) changed = true;
    EXPECT_TRUE(changed);
}

TEST(NoiseAugTest, GaussianNoiseOnGrayPreservesType) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    std::mt19937 rng(42);
    Image<Gray> result = RandomGaussianNoise{}(img, rng);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(NoiseAugTest, GaussianNoiseInvalidStdDevThrows) {
    EXPECT_THROW(RandomGaussianNoise{}.std_dev(-1.0f, 10.0f), std::invalid_argument);
    EXPECT_THROW(RandomGaussianNoise{}.std_dev(20.0f, 10.0f), std::invalid_argument);
}

TEST(NoiseAugTest, GaussianNoiseBindRngPipelineOp) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomGaussianNoise{}.std_dev(0.0f, 0.0f).bind(rng);
    EXPECT_EQ(result.rows(), 10);
}

// ---- RandomSaltAndPepper ----

TEST(NoiseAugTest, SaltAndPepperPreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomSaltAndPepper{}(img, rng);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(NoiseAugTest, SaltAndPepperAtP1AllPixelsAreExtremes) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    std::mt19937 rng(42);
    Image<Gray> result = RandomSaltAndPepper{}.p(1.0f)(img, rng);
    for (int r = 0; r < 10; ++r)
        for (int c = 0; c < 10; ++c) {
            uchar v = result.mat().at<uchar>(r, c);
            EXPECT_TRUE(v == 0 || v == 255) << "pixel(" << r << "," << c << ")=" << (int)v;
        }
}

TEST(NoiseAugTest, SaltAndPepperAtP0LeavesImageUnchanged) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    std::mt19937 rng(42);
    Image<Gray> result = RandomSaltAndPepper{}.p(0.0f)(img, rng);
    for (int r = 0; r < 10; ++r)
        for (int c = 0; c < 10; ++c)
            EXPECT_EQ(result.mat().at<uchar>(r, c), 100);
}

TEST(NoiseAugTest, SaltAndPepperInvalidPThrows) {
    EXPECT_THROW(RandomSaltAndPepper{}.p(-0.1f), std::invalid_argument);
    EXPECT_THROW(RandomSaltAndPepper{}.p(1.1f),  std::invalid_argument);
}

TEST(NoiseAugTest, SaltAndPepperBindRngPipelineOp) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomSaltAndPepper{}.p(0.0f).bind(rng);
    EXPECT_EQ(result.rows(), 10);
}
