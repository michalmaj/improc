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
    cv::Mat h = CalcHist{}(make_gray_rand(32, 32));
    EXPECT_EQ(h.rows, 256);
    EXPECT_EQ(h.cols, 1);
    EXPECT_EQ(h.type(), CV_32FC1);
}

TEST(CalcHistTest, GraySumEqualsPixelCount) {
    auto img = make_gray_rand(16, 16);
    cv::Mat h = CalcHist{}(img);
    double total = cv::sum(h)[0];
    EXPECT_NEAR(total, static_cast<double>(16 * 16), 1.0);
}

TEST(CalcHistTest, GrayAllBinsNonNegative) {
    cv::Mat h = CalcHist{}(make_gray_rand(32, 32));
    double mn;
    cv::minMaxLoc(h, &mn);
    EXPECT_GE(mn, 0.0);
}

TEST(CalcHistTest, BGROutputShape) {
    cv::Mat h = CalcHist{}(make_bgr_rand(32, 32));
    EXPECT_EQ(h.rows, 3 * 256);
    EXPECT_EQ(h.cols, 1);
    EXPECT_EQ(h.type(), CV_32FC1);
}

TEST(CalcHistTest, BGRSumEqualsThreeTimesPixelCount) {
    auto img = make_bgr_rand(16, 16);
    cv::Mat h = CalcHist{}(img);
    double total = cv::sum(h)[0];
    EXPECT_NEAR(total, static_cast<double>(3 * 16 * 16), 1.0);
}

TEST(CalcHistTest, CustomBinsChangesOutputRows) {
    cv::Mat h = CalcHist{}.bins(128)(make_gray_rand(32, 32));
    EXPECT_EQ(h.rows, 128);
    EXPECT_EQ(h.cols, 1);
}

// --- CompareHist ---

TEST(CompareHistTest, DefaultConstruction) {
    EXPECT_NO_THROW(CompareHist{});
}

TEST(CompareHistTest, IdenticalHistogramsScoreNearOne) {
    cv::Mat h = CalcHist{}(make_gray_rand(64, 64));
    double score = CompareHist{}(h, h);
    EXPECT_NEAR(score, 1.0, 1e-5);
}

// --- Single-bin test ---

TEST(CalcHistTest, ConstantGrayImageSingleNonZeroBin) {
    auto img = make_gray_solid(8, 8, 42);
    cv::Mat h = CalcHist{}(img);
    int nonzero = cv::countNonZero(h);
    EXPECT_EQ(nonzero, 1);
    EXPECT_NEAR(h.at<float>(42, 0), 64.0f, 1e-3f);
}
