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

// ── Convolve ──────────────────────────────────────────────────────────────────

TEST(ConvolveTest, IdentityKernelPreservesImage) {
    cv::Mat m(50, 50, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(m);
    cv::Mat kernel = (cv::Mat_<float>(1, 1) << 1.0f);
    auto result = img | Convolve(kernel);
    cv::Mat diff;
    cv::absdiff(result.mat(), m, diff);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

TEST(ConvolveTest, EmptyKernelThrows) {
    Image<Gray> img(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    EXPECT_THROW(img | Convolve(cv::Mat{}), std::invalid_argument);
}

TEST(ConvolveTest, FluentSettersReturnThis) {
    cv::Mat k = cv::Mat::eye(3, 3, CV_32F);
    Convolve op(k);
    EXPECT_EQ(&op.anchor({-1,-1}).delta(0.0).border(cv::BORDER_REFLECT_101), &op);
}

TEST(ConvolveTest, AveragingKernelBlursUniformImage) {
    cv::Mat m(50, 50, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(m);
    cv::Mat k = cv::Mat::ones(3, 3, CV_32F) / 9.0f;
    auto result = img | Convolve(k);
    // Uniform image convolved with averaging kernel stays uniform
    cv::Mat diff;
    cv::absdiff(result.mat(), m, diff);
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

// ── ConvertScaleAbs ───────────────────────────────────────────────────────────

TEST(ConvertScaleAbsTest, DefaultConstruction) {
    EXPECT_NO_THROW(ConvertScaleAbs{});
}

TEST(ConvertScaleAbsTest, FluentSettersReturnThis) {
    ConvertScaleAbs op;
    EXPECT_EQ(&op.alpha(2.0).beta(10.0), &op);
}

TEST(ConvertScaleAbsTest, CV16SPositiveValuesToCV8U) {
    // Simulate dx from Sobel: CV_16S with value 100 → scaled abs → 100 in CV_8U
    cv::Mat src(1, 1, CV_16S, cv::Scalar(100));
    auto result = ConvertScaleAbs{}(src);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 100u);
}

TEST(ConvertScaleAbsTest, NegativeValuesToAbsValue) {
    cv::Mat src(1, 1, CV_16S, cv::Scalar(-80));
    auto result = ConvertScaleAbs{}(src);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 80u);
}

TEST(ConvertScaleAbsTest, ScaleAlphaApplied) {
    cv::Mat src(1, 1, CV_16S, cv::Scalar(50));
    auto result = ConvertScaleAbs{}.alpha(2.0)(src);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 100u);
}

// ── Add ───────────────────────────────────────────────────────────────────────

TEST(AddTest, KnownScalarAddition) {
    cv::Mat m(1, 1, CV_8UC1, cv::Scalar(100));
    cv::Mat other(1, 1, CV_8UC1, cv::Scalar(50));
    Image<Gray> img(m);
    auto result = img | Add(other);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 150u);
}

TEST(AddTest, SaturationAtMax) {
    cv::Mat m(1, 1, CV_8UC1, cv::Scalar(200));
    cv::Mat other(1, 1, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(m);
    auto result = img | Add(other);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 255u);  // saturates at 255
}

TEST(AddTest, SizeMismatchThrows) {
    Image<Gray> img(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    EXPECT_THROW(img | Add(cv::Mat(60, 60, CV_8UC1)), std::invalid_argument);
}

// ── Subtract ──────────────────────────────────────────────────────────────────

TEST(SubtractTest, KnownScalarSubtraction) {
    cv::Mat m(1, 1, CV_8UC1, cv::Scalar(100));
    cv::Mat other(1, 1, CV_8UC1, cv::Scalar(40));
    Image<Gray> img(m);
    auto result = img | Subtract(other);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 60u);
}

TEST(SubtractTest, SaturationAtZero) {
    cv::Mat m(1, 1, CV_8UC1, cv::Scalar(50));
    cv::Mat other(1, 1, CV_8UC1, cv::Scalar(100));
    Image<Gray> img(m);
    auto result = img | Subtract(other);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 0u);  // saturates at 0
}

TEST(SubtractTest, SizeMismatchThrows) {
    Image<Gray> img(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    EXPECT_THROW(img | Subtract(cv::Mat(60, 60, CV_8UC1)), std::invalid_argument);
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
