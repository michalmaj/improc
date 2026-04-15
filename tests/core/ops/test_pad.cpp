// tests/core/ops/test_pad.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/pad.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// ---- Pad ----

TEST(PadTest, ConstantPadAddsCorrectRows) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    Image<BGR> result = Pad{}.top(2).bottom(3)(img);
    EXPECT_EQ(result.rows(), 15);  // 10 + 2 + 3
    EXPECT_EQ(result.cols(), 10);
}

TEST(PadTest, ConstantPadAddsCorrectCols) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    Image<BGR> result = Pad{}.left(4).right(1)(img);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 15);  // 10 + 4 + 1
}

TEST(PadTest, ReflectModeCorrectSize) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    Image<BGR> result = Pad{}.top(2).left(2).mode(PadMode::Reflect)(img);
    EXPECT_EQ(result.rows(), 12);
    EXPECT_EQ(result.cols(), 12);
}

TEST(PadTest, ReplicateModeCorrectSize) {
    cv::Mat mat(10, 10, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(mat);
    Image<Gray> result = Pad{}.bottom(5).right(5).mode(PadMode::Replicate)(img);
    EXPECT_EQ(result.rows(), 15);
    EXPECT_EQ(result.cols(), 15);
}

TEST(PadTest, AllZeroSidesThrows) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    EXPECT_THROW(Pad{}(img), std::invalid_argument);
}

TEST(PadTest, NegativeTopThrows) {
    EXPECT_THROW(Pad{}.top(-1), std::invalid_argument);
}

TEST(PadTest, NegativeLeftThrows) {
    EXPECT_THROW(Pad{}.left(-3), std::invalid_argument);
}

TEST(PadTest, PreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    Image<BGR> result = Pad{}.top(1).bottom(1).left(1).right(1)(img);
    EXPECT_EQ(result.rows(), 12);
    EXPECT_EQ(result.cols(), 12);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(PadTest, PipelineOp) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    Image<BGR> result = img | Pad{}.top(5);
    EXPECT_EQ(result.rows(), 15);
}

TEST(PadTest, ConstantPadFillsZeroByDefault) {
    // Default value_ is {0,0,0,0} — top-left pixel of padded area should be black
    cv::Mat mat(4, 4, CV_8UC1, cv::Scalar(200));
    Image<Gray> img(mat);
    Image<Gray> result = Pad{}.top(2)(img);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 0);  // top padding row, column 0
}

// ---- PadToSquare ----

TEST(PadTest, PadToSquareWideImageProducesSquare) {
    cv::Mat mat(6, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    Image<BGR> result = PadToSquare{}(img);
    EXPECT_EQ(result.rows(), result.cols());
    EXPECT_EQ(result.cols(), 10);
}

TEST(PadTest, PadToSquareTallImageProducesSquare) {
    cv::Mat mat(10, 6, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    Image<BGR> result = PadToSquare{}(img);
    EXPECT_EQ(result.rows(), result.cols());
    EXPECT_EQ(result.rows(), 10);
}

TEST(PadTest, PadToSquareAlreadySquareReturnsClone) {
    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    Image<BGR> result = PadToSquare{}(img);
    EXPECT_EQ(result.rows(), 8);
    EXPECT_EQ(result.cols(), 8);
}

TEST(PadTest, PadToSquarePipelineOp) {
    cv::Mat mat(6, 10, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(mat);
    Image<Gray> result = img | PadToSquare{};
    EXPECT_EQ(result.rows(), result.cols());
}
