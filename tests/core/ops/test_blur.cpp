// tests/core/ops/test_blur.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include <opencv2/core.hpp>
#include "improc/core/ops/blur.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(BlurTest, GaussianBlurDefaultPreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    Image<BGR> result = GaussianBlur{}(img);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(BlurTest, GaussianBlurKernel5PreservesSize) {
    cv::Mat mat(20, 20, CV_8UC3, cv::Scalar(50, 50, 50));
    Image<BGR> img(mat);
    Image<BGR> result = GaussianBlur{}.kernel_size(5)(img);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(BlurTest, GaussianBlurOnGrayPreservesType) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(mat);
    Image<Gray> result = GaussianBlur{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(BlurTest, GaussianBlurEvenKernelThrows) {
    EXPECT_THROW(GaussianBlur{}.kernel_size(4), improc::ParameterError);
}

TEST(BlurTest, GaussianBlurZeroKernelThrows) {
    EXPECT_THROW(GaussianBlur{}.kernel_size(0), improc::ParameterError);
}

TEST(BlurTest, GaussianBlurNegativeSigmaThrows) {
    EXPECT_THROW(GaussianBlur{}.sigma(-1.0), improc::ParameterError);
}

TEST(BlurTest, GaussianBlurPipelineOp) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    Image<BGR> result = img | GaussianBlur{}.kernel_size(3);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
}

TEST(BlurTest, MedianBlurDefaultPreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = MedianBlur{}(img);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(BlurTest, MedianBlurEvenKernelThrows) {
    EXPECT_THROW(MedianBlur{}.kernel_size(4), improc::ParameterError);
}

TEST(BlurTest, MedianBlurZeroKernelThrows) {
    EXPECT_THROW(MedianBlur{}.kernel_size(0), improc::ParameterError);
}

TEST(BlurTest, MedianBlurPipelineOp) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = img | MedianBlur{}.kernel_size(3);
    EXPECT_EQ(result.rows(), 10);
}

TEST(BlurTest, GaussianBlurActuallySmooths) {
    cv::Mat mat(5, 5, CV_8UC1, cv::Scalar(0));
    mat.at<uchar>(2, 2) = 255;  // single hot pixel
    Image<Gray> img(mat);
    Image<Gray> result = GaussianBlur{}.kernel_size(5)(img);
    EXPECT_LT(result.mat().at<uchar>(2, 2), 255);  // center should be smoothed
    EXPECT_GT(result.mat().at<uchar>(2, 3), 0);    // neighbor should pick up value
}

TEST(BlurTest, GaussianBlurNegativeKernelThrows) {
    EXPECT_THROW(GaussianBlur{}.kernel_size(-1), improc::ParameterError);
}

TEST(BlurTest, MedianBlurNegativeKernelThrows) {
    EXPECT_THROW(MedianBlur{}.kernel_size(-3), improc::ParameterError);
}
