// tests/core/ops/test_warp_affine.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/warp_affine.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;

TEST(WarpAffineTest, NoMatrixSetThrows) {
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100)));
    EXPECT_THROW(WarpAffine{}(img), improc::ParameterError);
}

TEST(WarpAffineTest, WrongShapeThrows) {
    cv::Mat bad(3, 3, CV_64F, cv::Scalar(1.0));
    EXPECT_THROW(WarpAffine{}.matrix(bad), improc::ParameterError);
}

TEST(WarpAffineTest, WrongTypeThrows) {
    cv::Mat bad(2, 3, CV_8U, cv::Scalar(1));
    EXPECT_THROW(WarpAffine{}.matrix(bad), improc::ParameterError);
}

TEST(WarpAffineTest, IdentityPreservesDimensions) {
    cv::Mat M = cv::Mat::eye(2, 3, CV_64F);
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100)));
    auto result = WarpAffine{}.matrix(M)(img);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

TEST(WarpAffineTest, CustomSizeApplied) {
    cv::Mat M = cv::Mat::eye(2, 3, CV_64F);
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100)));
    auto result = WarpAffine{}.matrix(M).width(128).height(96)(img);
    EXPECT_EQ(result.rows(), 96);
    EXPECT_EQ(result.cols(), 128);
}

TEST(WarpAffineTest, GrayFormatPreservesType) {
    cv::Mat M = cv::Mat::eye(2, 3, CV_64F);
    Image<Gray> img(cv::Mat(64, 64, CV_8UC1, cv::Scalar(128)));
    auto result = WarpAffine{}.matrix(M)(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(WarpAffineTest, PipelineFormComposes) {
    cv::Mat M = cv::Mat::eye(2, 3, CV_64F);
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100)));
    auto result = img | WarpAffine{}.matrix(M);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

TEST(WarpAffineTest, TranslationShiftsContent) {
    cv::Mat M = cv::Mat::eye(2, 3, CV_64F);
    M.at<double>(0, 2) = 10.0; // tx = 10
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(200, 100, 50)));
    auto result = WarpAffine{}.matrix(M)(img);
    cv::Vec3b topleft = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_EQ(topleft[0], 0);
    EXPECT_EQ(topleft[1], 0);
    EXPECT_EQ(topleft[2], 0);
}

TEST(WarpAffineTest, ZeroWidthThrows) {
    EXPECT_THROW(WarpAffine{}.width(0), improc::ParameterError);
}

TEST(WarpAffineTest, NegativeHeightThrows) {
    EXPECT_THROW(WarpAffine{}.height(-1), improc::ParameterError);
}
