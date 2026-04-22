// tests/core/ops/test_weighted_blend.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;
using improc::ParameterError;

TEST(WeightedBlendTest, AlphaOneReturnsImg1) {
    cv::Mat m1(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat m2(10, 10, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img1(m1);
    Image<BGR> img2(m2);
    Image<BGR> result = img1 | WeightedBlend<BGR>{img2}.alpha(1.0);
    double mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(mean, 100.0, 2.0);
}

TEST(WeightedBlendTest, AlphaZeroReturnsImg2) {
    cv::Mat m1(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat m2(10, 10, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img1(m1);
    Image<BGR> img2(m2);
    Image<BGR> result = img1 | WeightedBlend<BGR>{img2}.alpha(0.0);
    double mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(mean, 200.0, 2.0);
}

TEST(WeightedBlendTest, AlphaHalfReturnsMean) {
    cv::Mat m1(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat m2(10, 10, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img1(m1);
    Image<BGR> img2(m2);
    Image<BGR> result = img1 | WeightedBlend<BGR>{img2}.alpha(0.5);
    double mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(mean, 150.0, 2.0);
}

TEST(WeightedBlendTest, AlphaBelowZeroThrows) {
    cv::Mat m2(10, 10, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img2(m2);
    EXPECT_THROW(WeightedBlend<BGR>{img2}.alpha(-0.1), ParameterError);
}

TEST(WeightedBlendTest, AlphaAboveOneThrows) {
    cv::Mat m2(10, 10, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img2(m2);
    EXPECT_THROW(WeightedBlend<BGR>{img2}.alpha(1.1), ParameterError);
}

TEST(WeightedBlendTest, MismatchedSizesThrow) {
    cv::Mat m1(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat m2(5, 5, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img1(m1);
    Image<BGR> img2(m2);
    EXPECT_THROW(img1 | WeightedBlend<BGR>{img2}, ParameterError);
}

TEST(WeightedBlendTest, WorksOnGray) {
    cv::Mat m1(10, 10, CV_8UC1, cv::Scalar(100));
    cv::Mat m2(10, 10, CV_8UC1, cv::Scalar(200));
    Image<Gray> img1(m1);
    Image<Gray> img2(m2);
    Image<Gray> result = img1 | WeightedBlend<Gray>{img2}.alpha(0.5);
    double mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(mean, 150.0, 2.0);
}

TEST(WeightedBlendTest, WorksOnBGR) {
    cv::Mat m1(10, 10, CV_8UC3, cv::Scalar(80, 80, 80));
    cv::Mat m2(10, 10, CV_8UC3, cv::Scalar(180, 180, 180));
    Image<BGR> img1(m1);
    Image<BGR> img2(m2);
    Image<BGR> result = img1 | WeightedBlend<BGR>{img2}.alpha(0.5);
    double mean = cv::mean(result.mat())[0];
    EXPECT_NEAR(mean, 130.0, 2.0);
}
