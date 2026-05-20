// tests/core/ops/test_arithmetic.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/arithmetic.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// ── AbsDiff ───────────────────────────────────────────────────────────────────

TEST(AbsDiffTest, SameImageGivesZeroOutput) {
    cv::Mat m(50, 50, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(m);
    auto result = img | AbsDiff(m);
    EXPECT_EQ(cv::countNonZero(result.mat()), 0);
}

TEST(AbsDiffTest, KnownScalarDifference) {
    cv::Mat m1(1, 1, CV_8UC1, cv::Scalar(200));
    cv::Mat m2(1, 1, CV_8UC1, cv::Scalar(150));
    Image<Gray> img(m1);
    auto result = img | AbsDiff(m2);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 50u);
}

TEST(AbsDiffTest, SizeMismatchThrows) {
    Image<Gray> img(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    EXPECT_THROW(img | AbsDiff(cv::Mat(60, 60, CV_8UC1, cv::Scalar(0))), std::invalid_argument);
}

TEST(AbsDiffTest, TypeMismatchThrows) {
    Image<Gray> img(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    EXPECT_THROW(img | AbsDiff(cv::Mat(50, 50, CV_8UC3)), std::invalid_argument);
}

// ── BitwiseAnd ────────────────────────────────────────────────────────────────

TEST(BitwiseAndTest, AndWithZeroGivesZero) {
    cv::Mat m(50, 50, CV_8UC1, cv::Scalar(255));
    cv::Mat z(50, 50, CV_8UC1, cv::Scalar(0));
    Image<Gray> img(m);
    auto result = img | BitwiseAnd(z);
    EXPECT_EQ(cv::countNonZero(result.mat()), 0);
}

TEST(BitwiseAndTest, AndWith255IsIdentity) {
    cv::Mat m(50, 50, CV_8UC1, cv::Scalar(42));
    cv::Mat ones(50, 50, CV_8UC1, cv::Scalar(255));
    Image<Gray> img(m);
    auto result = img | BitwiseAnd(ones);
    cv::Mat diff;
    cv::absdiff(result.mat(), m, diff);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

TEST(BitwiseAndTest, SizeMismatchThrows) {
    Image<Gray> img(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    EXPECT_THROW(img | BitwiseAnd(cv::Mat(60, 60, CV_8UC1, cv::Scalar(0))), std::invalid_argument);
}

// ── BitwiseOr ─────────────────────────────────────────────────────────────────

TEST(BitwiseOrTest, OrWithZeroIsIdentity) {
    cv::Mat m(50, 50, CV_8UC1, cv::Scalar(42));
    cv::Mat z(50, 50, CV_8UC1, cv::Scalar(0));
    Image<Gray> img(m);
    auto result = img | BitwiseOr(z);
    cv::Mat diff;
    cv::absdiff(result.mat(), m, diff);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

TEST(BitwiseOrTest, OrWith255GivesAllOnes) {
    cv::Mat m(50, 50, CV_8UC1, cv::Scalar(0));
    cv::Mat ones(50, 50, CV_8UC1, cv::Scalar(255));
    Image<Gray> img(m);
    auto result = img | BitwiseOr(ones);
    EXPECT_EQ(cv::countNonZero(result.mat()), 50 * 50);
}

TEST(BitwiseOrTest, SizeMismatchThrows) {
    Image<Gray> img(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    EXPECT_THROW(img | BitwiseOr(cv::Mat(60, 60, CV_8UC1, cv::Scalar(0))), std::invalid_argument);
}

// ── BitwiseNot ────────────────────────────────────────────────────────────────

TEST(BitwiseNotTest, InvertsAllBits) {
    cv::Mat m(1, 1, CV_8UC1, cv::Scalar(0xFF));
    Image<Gray> img(m);
    auto result = img | BitwiseNot{};
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 0u);
}

TEST(BitwiseNotTest, DoubleInvertIsIdentity) {
    cv::Mat m(50, 50, CV_8UC1, cv::Scalar(123));
    Image<Gray> img(m);
    auto result = img | BitwiseNot{} | BitwiseNot{};
    cv::Mat diff;
    cv::absdiff(result.mat(), m, diff);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

// ── Pipeline syntax ───────────────────────────────────────────────────────────

TEST(ArithmeticTest, AllOpsPipelineSyntax) {
    cv::Mat m(10, 10, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(m);
    auto r1 = img | AbsDiff(m);
    auto r2 = img | BitwiseAnd(m);
    auto r3 = img | BitwiseOr(m);
    auto r4 = img | BitwiseNot{};
    EXPECT_EQ(r1.rows(), 10);
    EXPECT_EQ(r2.rows(), 10);
    EXPECT_EQ(r3.rows(), 10);
    EXPECT_EQ(r4.rows(), 10);
}
