// tests/core/ops/test_unsharp_mask.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/unsharp_mask.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;

TEST(UnsharpMaskTest, ZeroSigmaThrows) {
    EXPECT_THROW(UnsharpMask{}.sigma(0.0), improc::ParameterError);
}

TEST(UnsharpMaskTest, NegativeSigmaThrows) {
    EXPECT_THROW(UnsharpMask{}.sigma(-1.0), improc::ParameterError);
}

TEST(UnsharpMaskTest, ZeroStrengthThrows) {
    EXPECT_THROW(UnsharpMask{}.strength(0.0), improc::ParameterError);
}

TEST(UnsharpMaskTest, NegativeStrengthThrows) {
    EXPECT_THROW(UnsharpMask{}.strength(-0.5), improc::ParameterError);
}

TEST(UnsharpMaskTest, PreservesSize) {
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100)));
    auto result = UnsharpMask{}(img);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

TEST(UnsharpMaskTest, PreservesType) {
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100)));
    auto result = UnsharpMask{}.sigma(1.0).strength(0.5)(img);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(UnsharpMaskTest, GrayImagePreservesType) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(128)));
    auto result = UnsharpMask{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 32);
}

TEST(UnsharpMaskTest, PipelineFormComposes) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(80, 80, 80)));
    auto result = img | UnsharpMask{}.sigma(1.0).strength(1.0);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}
