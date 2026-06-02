// tests/core/ops/test_analysis_v080.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;

// ── IntegralImage ─────────────────────────────────────────────────────────────

TEST(IntegralImageTest, AllOnesGivesCorrectSum) {
    Image<Gray> img(cv::Mat(5, 5, CV_8UC1, cv::Scalar(1)));
    auto r = IntegralImage{}(img);
    // Integral image is (rows+1)×(cols+1). Sum at (5,5) = total pixel count = 25.
    EXPECT_EQ(r.sum.at<int>(5, 5), 25);
}

TEST(IntegralImageTest, WithSqSumNotEmpty) {
    Image<Gray> img(cv::Mat(5, 5, CV_8UC1, cv::Scalar(2)));
    auto r = IntegralImage{}.with_sq_sum(true)(img);
    EXPECT_FALSE(r.sq_sum.empty());
    // sq_sum at (5,5) = 5*5 * 2^2 = 100
    EXPECT_NEAR(r.sq_sum.at<double>(5, 5), 100.0, 1e-6);
}

TEST(IntegralImageTest, WithoutSqSumIsEmpty) {
    Image<Gray> img(cv::Mat(5, 5, CV_8UC1, cv::Scalar(1)));
    auto r = IntegralImage{}(img);
    EXPECT_TRUE(r.sq_sum.empty());
}

// ── MinMaxLoc ─────────────────────────────────────────────────────────────────

TEST(MinMaxLocTest, FindsKnownMinAndMax) {
    cv::Mat m(100, 100, CV_8UC1, cv::Scalar(128));
    m.at<uchar>(10, 20) = 0;    // min at row=10, col=20 → cv::Point(20, 10)
    m.at<uchar>(30, 40) = 255;  // max at row=30, col=40 → cv::Point(40, 30)
    Image<Gray> img(m);
    auto r = MinMaxLoc{}(img);
    EXPECT_EQ(r.min_val, 0.0);
    EXPECT_EQ(r.max_val, 255.0);
    EXPECT_EQ(r.min_loc, cv::Point(20, 10));
    EXPECT_EQ(r.max_loc, cv::Point(40, 30));
}

TEST(MinMaxLocTest, WorksOnCvMat) {
    cv::Mat m(10, 10, CV_32FC1, cv::Scalar(1.0f));
    m.at<float>(3, 7) = -5.0f;
    auto r = MinMaxLoc{}(m);
    EXPECT_NEAR(r.min_val, -5.0, 1e-5);
    EXPECT_EQ(r.min_loc, cv::Point(7, 3));
}

// ── MeanStdDev ────────────────────────────────────────────────────────────────

TEST(MeanStdDevTest, UniformImageZeroStdDev) {
    Image<Gray> img(cv::Mat(100, 100, CV_8UC1, cv::Scalar(128)));
    auto r = MeanStdDev{}(img);
    EXPECT_NEAR(r.mean[0],   128.0, 0.5);
    EXPECT_NEAR(r.stddev[0],   0.0, 0.5);
}

TEST(MeanStdDevTest, BGRReturnsPerChannelStats) {
    Image<BGR> img(cv::Mat(100, 100, CV_8UC3, cv::Scalar(10, 20, 30)));
    auto r = MeanStdDev{}(img);
    EXPECT_NEAR(r.mean[0], 10.0, 0.5);
    EXPECT_NEAR(r.mean[1], 20.0, 0.5);
    EXPECT_NEAR(r.mean[2], 30.0, 0.5);
}

// ── CountNonZero ──────────────────────────────────────────────────────────────

TEST(CountNonZeroTest, HalfFilledMask) {
    cv::Mat m(100, 100, CV_8UC1, cv::Scalar(0));
    m.rowRange(0, 50).setTo(cv::Scalar(255));
    Image<Gray> img(m);
    EXPECT_EQ(CountNonZero{}(img), 5000u);
}

TEST(CountNonZeroTest, AllZeroGivesZero) {
    Image<Gray> img(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    EXPECT_EQ(CountNonZero{}(img), 0u);
}

// ── Reduce ────────────────────────────────────────────────────────────────────

TEST(ReduceTest, SumAlongRowsGivesColumnSums) {
    // Image where column 0 has value 10, rest 0
    cv::Mat m(100, 4, CV_8UC1, cv::Scalar(0));
    m.col(0).setTo(cv::Scalar(10));
    Image<Gray> img(m);
    // dim=0: reduce rows → single row of column sums (CV_32SC1)
    auto result = Reduce{}.dim(0)(img);
    EXPECT_EQ(result.rows, 1);
    EXPECT_EQ(result.cols, 4);
    EXPECT_EQ(result.at<int>(0, 0), 1000);  // col 0: 100 rows × 10 = 1000
    EXPECT_EQ(result.at<int>(0, 1), 0);
}

TEST(ReduceTest, SumAlongColsGivesRowSums) {
    cv::Mat m(4, 100, CV_8UC1, cv::Scalar(0));
    m.row(0).setTo(cv::Scalar(5));
    Image<Gray> img(m);
    // dim=1: reduce cols → single col of row sums
    auto result = Reduce{}.dim(1)(img);
    EXPECT_EQ(result.rows, 4);
    EXPECT_EQ(result.cols, 1);
    EXPECT_EQ(result.at<int>(0, 0), 500);  // row 0: 100 cols × 5 = 500
    EXPECT_EQ(result.at<int>(1, 0), 0);
}

TEST(ReduceTest, AvgOutputsFloat) {
    // All pixels = 10 → avg = 10.0f
    Image<Gray> img(cv::Mat(4, 4, CV_8UC1, cv::Scalar(10)));
    auto result = Reduce{}.op(ReduceOp::Avg).dim(0)(img);
    EXPECT_EQ(result.type(), CV_32FC1);
    EXPECT_NEAR(result.at<float>(0, 0), 10.0f, 0.1f);
}

TEST(ReduceTest, MaxAlongRows) {
    cv::Mat m(3, 3, CV_8UC1, cv::Scalar(5));
    m.at<uchar>(1, 2) = 200;
    Image<Gray> img(m);
    // Max/Min output has same dtype as input (CV_8UC1)
    auto result = Reduce{}.op(ReduceOp::Max).dim(0)(img);
    EXPECT_EQ(result.at<uchar>(0, 2), 200u);
    EXPECT_EQ(result.at<uchar>(0, 0), 5u);
}

TEST(ReduceTest, MinAlongCols) {
    cv::Mat m(3, 3, CV_8UC1, cv::Scalar(100));
    m.at<uchar>(2, 0) = 1;
    Image<Gray> img(m);
    auto result = Reduce{}.op(ReduceOp::Min).dim(1)(img);
    EXPECT_EQ(result.at<uchar>(2, 0), 1u);
    EXPECT_EQ(result.at<uchar>(0, 0), 100u);
}

TEST(ReduceTest, InvalidDimThrows) {
    Image<Gray> img(cv::Mat(4, 4, CV_8UC1, cv::Scalar(0)));
    EXPECT_THROW(Reduce{}.dim(2), improc::ParameterError);
    EXPECT_THROW(Reduce{}.dim(-1), improc::ParameterError);
}
