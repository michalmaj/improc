// tests/core/ops/test_morphology.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
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
    EXPECT_EQ(result.cols(), 10);
}

TEST(MorphologyTest, DilateZeroKernelThrows) {
    EXPECT_THROW(Dilate{}.kernel_size(0), improc::ParameterError);
}

TEST(MorphologyTest, DilateNegativeKernelThrows) {
    EXPECT_THROW(Dilate{}.kernel_size(-1), improc::ParameterError);
}

TEST(MorphologyTest, DilateZeroIterationsThrows) {
    EXPECT_THROW(Dilate{}.iterations(0), improc::ParameterError);
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
    EXPECT_THROW(Erode{}.kernel_size(0), improc::ParameterError);
}

TEST(MorphologyTest, ErodeNegativeKernelThrows) {
    EXPECT_THROW(Erode{}.kernel_size(-2), improc::ParameterError);
}

TEST(MorphologyTest, ErodeZeroIterationsThrows) {
    EXPECT_THROW(Erode{}.iterations(0), improc::ParameterError);
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

TEST(MorphologyTest, DilateEvenKernelThrows) {
    EXPECT_THROW(Dilate{}.kernel_size(4), improc::ParameterError);
}

TEST(MorphologyTest, ErodeEvenKernelThrows) {
    EXPECT_THROW(Erode{}.kernel_size(2), improc::ParameterError);
}

TEST(MorphologyTest, ErodeWithIterations2PreservesSize) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(200));
    Image<Gray> img(mat);
    Image<Gray> result = Erode{}.iterations(2)(img);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
}

TEST(MorphologyTest, MorphOpenDefaultPreservesSizeAndType) {
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(0));
    Image<Gray> img(mat);
    Image<Gray> result = MorphOpen{}(img);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(MorphologyTest, MorphOpenOnBGRPreservesType) {
    cv::Mat mat(20, 20, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> img(mat);
    Image<BGR> result = MorphOpen{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(MorphologyTest, MorphOpenRemovesSmallNoise) {
    // Open = erode then dilate. An isolated bright pixel on a dark background is removed.
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(0));
    mat.at<uchar>(10, 10) = 255;
    Image<Gray> img(mat);
    Image<Gray> result = img | MorphOpen{}.kernel_size(3);
    EXPECT_EQ(result.mat().at<uchar>(10, 10), 0);
}

TEST(MorphologyTest, MorphOpenWithEllipseShapePreservesSize) {
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = MorphOpen{}.shape(MorphShape::Ellipse)(img);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(MorphologyTest, MorphOpenWithIterations2PreservesSize) {
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = MorphOpen{}.iterations(2)(img);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(MorphologyTest, MorphOpenZeroKernelThrows) {
    EXPECT_THROW(MorphOpen{}.kernel_size(0), improc::ParameterError);
}

TEST(MorphologyTest, MorphOpenEvenKernelThrows) {
    EXPECT_THROW(MorphOpen{}.kernel_size(4), improc::ParameterError);
}

TEST(MorphologyTest, MorphOpenZeroIterationsThrows) {
    EXPECT_THROW(MorphOpen{}.iterations(0), improc::ParameterError);
}

TEST(MorphologyTest, MorphOpenPipelineOp) {
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = img | MorphOpen{}.kernel_size(3);
    EXPECT_EQ(result.rows(), 20);
}

TEST(MorphologyTest, MorphCloseDefaultPreservesSizeAndType) {
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(255));
    Image<Gray> img(mat);
    Image<Gray> result = MorphClose{}(img);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(MorphologyTest, MorphCloseOnBGRPreservesType) {
    cv::Mat mat(20, 20, CV_8UC3, cv::Scalar(255, 255, 255));
    Image<BGR> img(mat);
    Image<BGR> result = MorphClose{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(MorphologyTest, MorphCloseFillsSmallHoles) {
    // Close = dilate then erode. An isolated dark pixel on a bright background is filled.
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(255));
    mat.at<uchar>(10, 10) = 0;
    Image<Gray> img(mat);
    Image<Gray> result = img | MorphClose{}.kernel_size(3);
    EXPECT_EQ(result.mat().at<uchar>(10, 10), 255);
}

TEST(MorphologyTest, MorphCloseWithEllipseShapePreservesSize) {
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(200));
    Image<Gray> img(mat);
    Image<Gray> result = MorphClose{}.shape(MorphShape::Ellipse)(img);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(MorphologyTest, MorphCloseWithIterations2PreservesSize) {
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(200));
    Image<Gray> img(mat);
    Image<Gray> result = MorphClose{}.iterations(2)(img);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(MorphologyTest, MorphCloseZeroKernelThrows) {
    EXPECT_THROW(MorphClose{}.kernel_size(0), improc::ParameterError);
}

TEST(MorphologyTest, MorphCloseEvenKernelThrows) {
    EXPECT_THROW(MorphClose{}.kernel_size(4), improc::ParameterError);
}

TEST(MorphologyTest, MorphCloseZeroIterationsThrows) {
    EXPECT_THROW(MorphClose{}.iterations(0), improc::ParameterError);
}

TEST(MorphologyTest, MorphClosePipelineOp) {
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(200));
    Image<Gray> img(mat);
    Image<Gray> result = img | MorphClose{}.kernel_size(3);
    EXPECT_EQ(result.rows(), 20);
}

// ── MorphGradient ─────────────────────────────────────────────────────────────

TEST(MorphGradientTest, GrayDefaultPreservesSizeAndType) {
    Image<Gray> img(cv::Mat(20, 20, CV_8UC1, cv::Scalar(128)));
    auto result = MorphGradient{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(MorphGradientTest, BGRDefaultPreservesSizeAndType) {
    Image<BGR> img(cv::Mat(20, 20, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = MorphGradient{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(MorphGradientTest, GrayUniformImageProducesZeroEdges) {
    // Gradient = dilate − erode; on a flat image both are the same, so result is 0
    Image<Gray> img(cv::Mat(20, 20, CV_8UC1, cv::Scalar(128)));
    auto result = MorphGradient{}(img);
    // Interior pixels should be 0; check a central pixel
    EXPECT_EQ(result.mat().at<uchar>(10, 10), 0);
}

TEST(MorphGradientTest, GrayPipelineSyntax) {
    Image<Gray> img(cv::Mat(20, 20, CV_8UC1, cv::Scalar(100)));
    auto result = img | MorphGradient{};
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(MorphGradientTest, KernelSizeEvenThrows) {
    EXPECT_THROW(MorphGradient{}.kernel_size(4), improc::ParameterError);
}

TEST(MorphGradientTest, KernelSizeZeroThrows) {
    EXPECT_THROW(MorphGradient{}.kernel_size(0), improc::ParameterError);
}

TEST(MorphGradientTest, KernelSizeNegativeThrows) {
    EXPECT_THROW(MorphGradient{}.kernel_size(-1), improc::ParameterError);
}

// ── TopHat ────────────────────────────────────────────────────────────────────

TEST(TopHatTest, GrayDefaultPreservesSizeAndType) {
    Image<Gray> img(cv::Mat(20, 20, CV_8UC1, cv::Scalar(128)));
    auto result = TopHat{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(TopHatTest, BGRDefaultPreservesSizeAndType) {
    Image<BGR> img(cv::Mat(20, 20, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = TopHat{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(TopHatTest, GrayUniformImageProducesZeroOutput) {
    // TopHat = src − MorphOpen; on a flat image MorphOpen = src, so result is 0
    Image<Gray> img(cv::Mat(20, 20, CV_8UC1, cv::Scalar(128)));
    auto result = TopHat{}(img);
    EXPECT_EQ(result.mat().at<uchar>(10, 10), 0);
}

TEST(TopHatTest, GrayPipelineSyntax) {
    Image<Gray> img(cv::Mat(20, 20, CV_8UC1, cv::Scalar(100)));
    auto result = img | TopHat{};
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(TopHatTest, KernelSizeEvenThrows) {
    EXPECT_THROW(TopHat{}.kernel_size(4), improc::ParameterError);
}

TEST(TopHatTest, KernelSizeZeroThrows) {
    EXPECT_THROW(TopHat{}.kernel_size(0), improc::ParameterError);
}

// ── BlackHat ──────────────────────────────────────────────────────────────────

TEST(BlackHatTest, GrayDefaultPreservesSizeAndType) {
    Image<Gray> img(cv::Mat(20, 20, CV_8UC1, cv::Scalar(128)));
    auto result = BlackHat{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(BlackHatTest, BGRDefaultPreservesSizeAndType) {
    Image<BGR> img(cv::Mat(20, 20, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = BlackHat{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(BlackHatTest, GrayUniformImageProducesZeroOutput) {
    // BlackHat = MorphClose − src; on a flat image MorphClose = src, so result is 0
    Image<Gray> img(cv::Mat(20, 20, CV_8UC1, cv::Scalar(128)));
    auto result = BlackHat{}(img);
    EXPECT_EQ(result.mat().at<uchar>(10, 10), 0);
}

TEST(BlackHatTest, GrayPipelineSyntax) {
    Image<Gray> img(cv::Mat(20, 20, CV_8UC1, cv::Scalar(100)));
    auto result = img | BlackHat{};
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(BlackHatTest, KernelSizeEvenThrows) {
    EXPECT_THROW(BlackHat{}.kernel_size(4), improc::ParameterError);
}

TEST(BlackHatTest, KernelSizeZeroThrows) {
    EXPECT_THROW(BlackHat{}.kernel_size(0), improc::ParameterError);
}
