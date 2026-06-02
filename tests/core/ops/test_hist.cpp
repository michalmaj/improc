// tests/core/ops/test_hist.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/hist.hpp"

using namespace improc::core;

namespace {
Image<Gray> make_gray_rand(int rows, int cols) {
    cv::Mat m(rows, cols, CV_8UC1);
    cv::randu(m, 0, 256);
    return Image<Gray>(m);
}
Image<Gray> make_gray_solid(int rows, int cols, uint8_t val) {
    return Image<Gray>(cv::Mat(rows, cols, CV_8UC1, cv::Scalar(val)));
}
Image<BGR> make_bgr_rand(int rows, int cols) {
    cv::Mat m(rows, cols, CV_8UC3);
    cv::randu(m, 0, 256);
    return Image<BGR>(m);
}
} // namespace

// --- CalcHist ---

TEST(CalcHistTest, DefaultConstruction) {
    EXPECT_NO_THROW(CalcHist{});
}

TEST(CalcHistTest, FluentSetterReturnsThis) {
    CalcHist op;
    EXPECT_EQ(&op.bins(128), &op);
}

TEST(CalcHistTest, GrayOutputShape) {
    HistogramData h = CalcHist{}(make_gray_rand(32, 32));
    EXPECT_EQ(h.data.rows, 256);
    EXPECT_EQ(h.data.cols, 1);
    EXPECT_EQ(h.data.type(), CV_32FC1);
}

TEST(CalcHistTest, GraySumEqualsPixelCount) {
    auto img = make_gray_rand(16, 16);
    HistogramData h = CalcHist{}(img);
    double total = cv::sum(h.data)[0];
    EXPECT_NEAR(total, static_cast<double>(16 * 16), 0.5);
}

TEST(CalcHistTest, GrayAllBinsNonNegative) {
    HistogramData h = CalcHist{}(make_gray_rand(32, 32));
    double mn;
    cv::minMaxLoc(h.data, &mn);
    EXPECT_GE(mn, 0.0);
}

TEST(CalcHistTest, BGROutputShape) {
    HistogramData h = CalcHist{}(make_bgr_rand(32, 32));
    EXPECT_EQ(h.data.rows, 256);
    EXPECT_EQ(h.data.cols, 3);
    EXPECT_EQ(h.data.type(), CV_32FC1);
}

TEST(CalcHistTest, BGRSumEqualsThreeTimesPixelCount) {
    auto img = make_bgr_rand(16, 16);
    HistogramData h = CalcHist{}(img);
    double total = cv::sum(h.data)[0];
    EXPECT_NEAR(total, static_cast<double>(3 * 16 * 16), 0.5);
}

TEST(CalcHistTest, CustomBinsChangesOutputRows) {
    HistogramData h = CalcHist{}.bins(128)(make_gray_rand(32, 32));
    EXPECT_EQ(h.data.rows, 128);
    EXPECT_EQ(h.data.cols, 1);
}

// --- CompareHist ---

TEST(CompareHistTest, DefaultConstruction) {
    EXPECT_NO_THROW(CompareHist{});
}

TEST(CompareHistTest, IdenticalHistogramsScoreNearOne) {
    HistogramData h = CalcHist{}(make_gray_rand(64, 64));
    double score = CompareHist{}(h, h);
    EXPECT_NEAR(score, 1.0, 1e-5);
}

// --- Single-bin test ---

TEST(CalcHistTest, ConstantGrayImageSingleNonZeroBin) {
    auto img = make_gray_solid(8, 8, 42);
    HistogramData h = CalcHist{}(img);
    int nonzero = cv::countNonZero(h.data);
    EXPECT_EQ(nonzero, 1);
    EXPECT_NEAR(h.data.at<float>(42, 0), 64.0f, 1e-3f);
}

// --- HistogramData type tests ---

TEST(CalcHistTest, ReturnsHistogramDataForGray) {
    cv::Mat m(100, 100, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(m);
    HistogramData h = CalcHist{}(img);
    EXPECT_FALSE(h.empty());
    EXPECT_EQ(h.bins, 256);
    EXPECT_EQ(h.channels, 1);
}

TEST(CalcHistTest, ReturnsHistogramDataForBGR) {
    cv::Mat m(100, 100, CV_8UC3, cv::Scalar(50, 100, 150));
    Image<BGR> img(m);
    HistogramData h = CalcHist{}(img);
    EXPECT_FALSE(h.empty());
    EXPECT_EQ(h.bins, 256);
    EXPECT_EQ(h.channels, 3);
}

TEST(CalcHistTest, HistogramDataMetadataMatchesCustomSettings) {
    HistogramData h = CalcHist{}.bins(128).range(0.0f, 128.0f)(make_gray_rand(32, 32));
    EXPECT_EQ(h.bins, 128);
    EXPECT_FLOAT_EQ(h.range_min, 0.0f);
    EXPECT_FLOAT_EQ(h.range_max, 128.0f);
    EXPECT_EQ(h.channels, 1);
}

TEST(CompareHistTest, AcceptsHistogramData) {
    cv::Mat m(100, 100, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(m);
    HistogramData h = CalcHist{}(img);
    double corr = CompareHist{}(h, h);
    EXPECT_NEAR(corr, 1.0, 1e-5);
}
