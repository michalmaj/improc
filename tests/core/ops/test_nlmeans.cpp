// tests/core/ops/test_nlmeans.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/core/ops/nlmeans.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(NLMeansDenoisingTest, GrayDefaultPreservesSizeAndType) {
    Image<Gray> img(cv::Mat(64, 64, CV_8UC1, cv::Scalar(128)));
    auto result = NLMeansDenoising{}(img);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(NLMeansDenoisingTest, BGRDefaultPreservesSizeAndType) {
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(100, 120, 80)));
    auto result = NLMeansDenoising{}(img);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(NLMeansDenoisingTest, GrayReducesNoise) {
    // Generate noisy gray image via float arithmetic to avoid uchar clamping
    cv::Mat base(64, 64, CV_32FC1, cv::Scalar(128.0f));
    cv::Mat noise(64, 64, CV_32FC1);
    cv::randn(noise, 0.0f, 30.0f);
    cv::Mat noisy_f;
    cv::add(base, noise, noisy_f);
    cv::Mat noisy;
    noisy_f.convertTo(noisy, CV_8UC1);

    cv::Scalar mean_before, stddev_before;
    cv::meanStdDev(noisy, mean_before, stddev_before);

    Image<Gray> img(noisy);
    auto result = NLMeansDenoising{}.h(10.0f)(img);

    cv::Scalar mean_after, stddev_after;
    cv::meanStdDev(result.mat(), mean_after, stddev_after);

    EXPECT_LT(stddev_after[0], stddev_before[0]);
}

TEST(NLMeansDenoisingTest, BGRReducesNoise) {
    cv::Mat base(64, 64, CV_32FC3, cv::Scalar(100.0f, 120.0f, 80.0f));
    cv::Mat noise(64, 64, CV_32FC3);
    cv::randn(noise, 0.0f, 30.0f);
    cv::Mat noisy_f;
    cv::add(base, noise, noisy_f);
    cv::Mat noisy;
    noisy_f.convertTo(noisy, CV_8UC3);

    cv::Scalar mean_before, stddev_before;
    cv::meanStdDev(noisy, mean_before, stddev_before);

    Image<BGR> img(noisy);
    auto result = NLMeansDenoising{}.h(10.0f).h_color(10.0f)(img);

    cv::Scalar mean_after, stddev_after;
    cv::meanStdDev(result.mat(), mean_after, stddev_after);

    EXPECT_LT(stddev_after[0], stddev_before[0]);
}

TEST(NLMeansDenoisingTest, HZeroThrows) {
    EXPECT_THROW(NLMeansDenoising{}.h(0.0f), improc::ParameterError);
}

TEST(NLMeansDenoisingTest, HNegativeThrows) {
    EXPECT_THROW(NLMeansDenoising{}.h(-1.0f), improc::ParameterError);
}

TEST(NLMeansDenoisingTest, HColorZeroThrows) {
    EXPECT_THROW(NLMeansDenoising{}.h_color(0.0f), improc::ParameterError);
}

TEST(NLMeansDenoisingTest, TemplateWindowSizeEvenThrows) {
    EXPECT_THROW(NLMeansDenoising{}.template_window_size(4), improc::ParameterError);
}

TEST(NLMeansDenoisingTest, TemplateWindowSizeZeroThrows) {
    EXPECT_THROW(NLMeansDenoising{}.template_window_size(0), improc::ParameterError);
}

TEST(NLMeansDenoisingTest, SearchWindowSizeEvenThrows) {
    EXPECT_THROW(NLMeansDenoising{}.search_window_size(4), improc::ParameterError);
}

TEST(NLMeansDenoisingTest, HColorOnGrayDoesNotThrow) {
    // h_color setter always accepts valid positive values even when used with Gray
    Image<Gray> img(cv::Mat(64, 64, CV_8UC1, cv::Scalar(128)));
    EXPECT_NO_THROW({
        auto result = NLMeansDenoising{}.h(3.0f).h_color(3.0f)(img);
        (void)result;
    });
}

TEST(NLMeansDenoisingTest, GrayPipelineSyntax) {
    Image<Gray> img(cv::Mat(64, 64, CV_8UC1, cv::Scalar(128)));
    auto result = img | NLMeansDenoising{}.h(3.0f);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}
