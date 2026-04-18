// tests/core/ops/test_edge.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/core/ops/edge.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// ── SobelEdge ────────────────────────────────────────────────────────────────

TEST(SobelEdgeTest, InvalidKsizeThrows) {
    EXPECT_THROW(SobelEdge{}.ksize(2), improc::ParameterError);
    EXPECT_THROW(SobelEdge{}.ksize(4), improc::ParameterError);
    EXPECT_THROW(SobelEdge{}.ksize(0), improc::ParameterError);
}

TEST(SobelEdgeTest, GrayOutputIsGrayType) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(128)));
    auto result = SobelEdge{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}

TEST(SobelEdgeTest, BGRInputProducesGrayOutput) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = SobelEdge{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(SobelEdgeTest, UniformImageProducesZeroEdges) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(128)));
    auto result = SobelEdge{}(img);
    // Interior pixels of a flat image have zero gradient
    EXPECT_EQ(result.mat().at<uchar>(16, 16), 0);
}

TEST(SobelEdgeTest, DetectsVerticalEdge) {
    cv::Mat mat(32, 32, CV_8UC1, cv::Scalar(0));
    mat(cv::Rect(16, 0, 16, 32)) = 255;  // right half white
    Image<Gray> img(mat);
    auto result = SobelEdge{}(img);
    // The edge column should have non-zero gradient
    EXPECT_GT(static_cast<int>(result.mat().at<uchar>(16, 15)), 0);
}

TEST(SobelEdgeTest, PipelineFormGray) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(100)));
    auto result = img | SobelEdge{}.ksize(3);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(SobelEdgeTest, PipelineFormBGR) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = img | SobelEdge{};
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

// ── CannyEdge ────────────────────────────────────────────────────────────────

TEST(CannyEdgeTest, NegativeThreshold1Throws) {
    EXPECT_THROW(CannyEdge{}.threshold1(-1.0), improc::ParameterError);
}

TEST(CannyEdgeTest, NegativeThreshold2Throws) {
    EXPECT_THROW(CannyEdge{}.threshold2(-1.0), improc::ParameterError);
}

TEST(CannyEdgeTest, InvalidApertureSizeThrows) {
    EXPECT_THROW(CannyEdge{}.aperture_size(2), improc::ParameterError);
    EXPECT_THROW(CannyEdge{}.aperture_size(4), improc::ParameterError);
}

TEST(CannyEdgeTest, GrayOutputIsBinary) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(128)));
    auto result = CannyEdge{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}

TEST(CannyEdgeTest, BGRInputProducesGrayOutput) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = CannyEdge{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(CannyEdgeTest, UniformImageProducesNoEdges) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(128)));
    auto result = CannyEdge{}(img);
    EXPECT_EQ(cv::countNonZero(result.mat()), 0);
}

TEST(CannyEdgeTest, PipelineForm) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(100)));
    auto result = img | CannyEdge{}.threshold1(50).threshold2(150);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}
