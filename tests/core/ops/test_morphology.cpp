// tests/core/ops/test_morphology.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/morphology.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(MorphologyTest, DilateDefaultPreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = Dilate{}(img);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(MorphologyTest, DilateOnBGRPreservesType) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    Image<BGR> result = Dilate{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(MorphologyTest, DilateWithEllipseShapePreservesSize) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = Dilate{}.shape(MorphShape::Ellipse)(img);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
}

TEST(MorphologyTest, DilateWithIterations2PreservesSize) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = Dilate{}.iterations(2)(img);
    EXPECT_EQ(result.rows(), 10);
}

TEST(MorphologyTest, DilateZeroKernelThrows) {
    EXPECT_THROW(Dilate{}.kernel_size(0), std::invalid_argument);
}

TEST(MorphologyTest, DilateNegativeKernelThrows) {
    EXPECT_THROW(Dilate{}.kernel_size(-1), std::invalid_argument);
}

TEST(MorphologyTest, DilateZeroIterationsThrows) {
    EXPECT_THROW(Dilate{}.iterations(0), std::invalid_argument);
}

TEST(MorphologyTest, DilatePipelineOp) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = img | Dilate{}.kernel_size(3);
    EXPECT_EQ(result.rows(), 10);
}

TEST(MorphologyTest, DilateActuallyDilates) {
    cv::Mat mat(5, 5, CV_8UC1, cv::Scalar(0));
    mat.at<uchar>(2, 2) = 255;  // single bright pixel
    Image<Gray> img(mat);
    Image<Gray> result = Dilate{}.kernel_size(3)(img);
    // Neighbors should now be bright (dilation spreads bright pixels)
    EXPECT_GT(result.mat().at<uchar>(2, 3), 0);
    EXPECT_GT(result.mat().at<uchar>(3, 2), 0);
}

TEST(MorphologyTest, ErodeDefaultPreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(200));
    Image<Gray> img(mat);
    Image<Gray> result = Erode{}(img);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(MorphologyTest, ErodeZeroKernelThrows) {
    EXPECT_THROW(Erode{}.kernel_size(0), std::invalid_argument);
}

TEST(MorphologyTest, ErodeNegativeKernelThrows) {
    EXPECT_THROW(Erode{}.kernel_size(-2), std::invalid_argument);
}

TEST(MorphologyTest, ErodeZeroIterationsThrows) {
    EXPECT_THROW(Erode{}.iterations(0), std::invalid_argument);
}

TEST(MorphologyTest, ErodePipelineOp) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(200));
    Image<Gray> img(mat);
    Image<Gray> result = img | Erode{}.kernel_size(3);
    EXPECT_EQ(result.rows(), 10);
}

TEST(MorphologyTest, ErodeActuallyErodes) {
    cv::Mat mat(5, 5, CV_8UC1, cv::Scalar(255));
    mat.at<uchar>(2, 2) = 0;  // single dark pixel
    Image<Gray> img(mat);
    Image<Gray> result = Erode{}.kernel_size(3)(img);
    // Neighbors of dark pixel should darken (erosion spreads dark pixels)
    EXPECT_LT(result.mat().at<uchar>(2, 3), 255);
    EXPECT_LT(result.mat().at<uchar>(3, 2), 255);
}
