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

// ── LaplacianEdge ─────────────────────────────────────────────────────────────

TEST(LaplacianEdgeTest, GrayDefaultPreservesSizeAndType) {
    Image<Gray> img(cv::Mat(20, 20, CV_8UC1, cv::Scalar(128)));
    auto result = LaplacianEdge{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(LaplacianEdgeTest, BGRDefaultPreservesSizeAndType) {
    Image<BGR> img(cv::Mat(20, 20, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = LaplacianEdge{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 20);
    EXPECT_EQ(result.cols(), 20);
}

TEST(LaplacianEdgeTest, GrayDetectsEdges) {
    // Left half black, right half white — strong edge at the boundary
    cv::Mat mat(32, 32, CV_8UC1, cv::Scalar(0));
    mat(cv::Rect(16, 0, 16, 32)) = 255;
    Image<Gray> img(mat);
    auto result = LaplacianEdge{}.ksize(1)(img);
    // Pixels at the step edge (columns 15-16) should be non-zero
    bool has_edge = false;
    for (int r = 1; r < 31; ++r) {
        if (result.mat().at<uchar>(r, 15) > 0 || result.mat().at<uchar>(r, 16) > 0) {
            has_edge = true;
            break;
        }
    }
    EXPECT_TRUE(has_edge);
}

TEST(LaplacianEdgeTest, BGRDetectsEdges) {
    cv::Mat mat(32, 32, CV_8UC3, cv::Scalar(0, 0, 0));
    mat(cv::Rect(16, 0, 16, 32)) = cv::Scalar(255, 255, 255);
    Image<BGR> img(mat);
    auto result = LaplacianEdge{}.ksize(1)(img);
    bool has_edge = false;
    for (int r = 1; r < 31; ++r) {
        if (result.mat().at<uchar>(r, 15) > 0 || result.mat().at<uchar>(r, 16) > 0) {
            has_edge = true;
            break;
        }
    }
    EXPECT_TRUE(has_edge);
}

TEST(LaplacianEdgeTest, GrayPipelineSyntax) {
    Image<Gray> img(cv::Mat(20, 20, CV_8UC1, cv::Scalar(100)));
    auto result = img | LaplacianEdge{};
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(LaplacianEdgeTest, BGRPipelineSyntax) {
    Image<BGR> img(cv::Mat(20, 20, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = img | LaplacianEdge{}.ksize(3);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(LaplacianEdgeTest, KsizeEvenThrows) {
    EXPECT_THROW(LaplacianEdge{}.ksize(4), improc::ParameterError);
}

TEST(LaplacianEdgeTest, KsizeZeroThrows) {
    EXPECT_THROW(LaplacianEdge{}.ksize(0), improc::ParameterError);
}

TEST(LaplacianEdgeTest, ScaleZeroThrows) {
    EXPECT_THROW(LaplacianEdge{}.scale(0.0), improc::ParameterError);
}

TEST(LaplacianEdgeTest, ScaleNegativeThrows) {
    EXPECT_THROW(LaplacianEdge{}.scale(-1.0), improc::ParameterError);
}

TEST(LaplacianEdgeTest, KsizeNegativeThrows) {
    EXPECT_THROW(LaplacianEdge{}.ksize(-1), improc::ParameterError);
}

TEST(LaplacianEdgeTest, DeltaShiftsOutput) {
    // Flat image has zero Laplacian; delta=50 shifts output above zero
    Image<Gray> flat(cv::Mat(20, 20, CV_8UC1, cv::Scalar(128)));
    auto result = LaplacianEdge{}.delta(50.0)(flat);
    EXPECT_GT(cv::mean(result.mat())[0], 40.0);
}

// ── HarrisCorner ──────────────────────────────────────────────────────────────

TEST(HarrisCornerTest, GrayDefaultPreservesSizeAndType) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(100)));
    auto result = HarrisCorner{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}

TEST(HarrisCornerTest, BGRDefaultPreservesSizeAndType) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = HarrisCorner{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
}

TEST(HarrisCornerTest, GrayUniformImageNoCorners) {
    // Uniform image has zero Harris response everywhere; after NORM_MINMAX it stays 0
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(128)));
    auto result = HarrisCorner{}(img);
    EXPECT_EQ(cv::countNonZero(result.mat()), 0);
}

TEST(HarrisCornerTest, GrayPipelineSyntax) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(100)));
    auto result = img | HarrisCorner{};
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(HarrisCornerTest, BGRPipelineSyntax) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(80, 100, 120)));
    auto result = img | HarrisCorner{}.ksize(3);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(HarrisCornerTest, KsizeInvalidThrows) {
    EXPECT_THROW(HarrisCorner{}.ksize(2), improc::ParameterError);
    EXPECT_THROW(HarrisCorner{}.ksize(4), improc::ParameterError);
    EXPECT_THROW(HarrisCorner{}.ksize(1), improc::ParameterError);
}

TEST(HarrisCornerTest, KsizeValidDoesNotThrow) {
    EXPECT_NO_THROW(HarrisCorner{}.ksize(5));
    EXPECT_NO_THROW(HarrisCorner{}.ksize(7));
}

TEST(HarrisCornerTest, BlockSizeZeroThrows) {
    EXPECT_THROW(HarrisCorner{}.block_size(0), improc::ParameterError);
    EXPECT_THROW(HarrisCorner{}.block_size(-1), improc::ParameterError);
}

TEST(HarrisCornerTest, KZeroThrows) {
    EXPECT_THROW(HarrisCorner{}.k(0.0), improc::ParameterError);
    EXPECT_THROW(HarrisCorner{}.k(-0.01), improc::ParameterError);
}

TEST(HarrisCornerTest, KOneThrows) {
    EXPECT_THROW(HarrisCorner{}.k(1.0), improc::ParameterError);
    EXPECT_THROW(HarrisCorner{}.k(1.5), improc::ParameterError);
}

TEST(HarrisCornerTest, DetectsCornerOnWhiteSquare) {
    // White square on black background — corners of the square have strong Harris response
    cv::Mat mat(32, 32, CV_8UC1, cv::Scalar(0));
    mat(cv::Rect(8, 8, 16, 16)) = 255;
    Image<Gray> img(mat);
    auto result = HarrisCorner{}(img);
    // Corner at (8,8): should have non-zero response
    // Flat interior at (16,16): should have low response
    int corner_val  = result.mat().at<uchar>(8, 8);
    int interior_val = result.mat().at<uchar>(16, 16);
    EXPECT_GT(corner_val, interior_val);
}
