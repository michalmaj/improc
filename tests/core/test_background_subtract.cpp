// tests/core/test_background_subtract.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/background_subtract.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/core/image.hpp"

using namespace improc::core;

namespace {
Image<BGR> make_bgr(int rows, int cols, cv::Scalar colour = {100, 150, 200}) {
    return Image<BGR>(cv::Mat(rows, cols, CV_8UC3, colour));
}
} // namespace

// ── BackgroundSubtractMOG2 ───────────────────────────────────────────────────

TEST(BackgroundSubtractMOG2Test, DefaultConstructionDoesNotThrow) {
    EXPECT_NO_THROW(BackgroundSubtractMOG2{});
}

TEST(BackgroundSubtractMOG2Test, ReturnsCorrectSizeGrayMask) {
    BackgroundSubtractMOG2 sub;
    auto fg = sub(make_bgr(48, 64));
    EXPECT_EQ(fg.mat().rows, 48);
    EXPECT_EQ(fg.mat().cols, 64);
    EXPECT_EQ(fg.mat().type(), CV_8UC1);
}

TEST(BackgroundSubtractMOG2Test, PipelineOperatorSyntax) {
    BackgroundSubtractMOG2 sub;
    Image<Gray> fg = make_bgr(24, 32) | sub;
    EXPECT_EQ(fg.mat().rows, 24);
    EXPECT_EQ(fg.mat().cols, 32);
}

TEST(BackgroundSubtractMOG2Test, FluentSettersReturnThis) {
    BackgroundSubtractMOG2 sub;
    auto& ref1 = sub.history(300);
    auto& ref2 = sub.threshold(20.0);
    auto& ref3 = sub.detect_shadows(false);
    EXPECT_EQ(&ref1, &sub);
    EXPECT_EQ(&ref2, &sub);
    EXPECT_EQ(&ref3, &sub);
}

TEST(BackgroundSubtractMOG2Test, ModelUpdatesAcrossFrames) {
    BackgroundSubtractMOG2 sub;
    auto frame = make_bgr(24, 32);
    // Apply same frame repeatedly — model learns background
    for (int i = 0; i < 5; ++i) sub(frame);
    // Now feed a very different frame — should produce non-zero foreground
    auto different = Image<BGR>(cv::Mat(24, 32, CV_8UC3, cv::Scalar(0, 0, 0)));
    Image<Gray> fg = different | sub;
    EXPECT_GT(cv::countNonZero(fg.mat()), 0);
}

// ── BackgroundSubtractKNN ────────────────────────────────────────────────────

TEST(BackgroundSubtractKNNTest, DefaultConstructionDoesNotThrow) {
    EXPECT_NO_THROW(BackgroundSubtractKNN{});
}

TEST(BackgroundSubtractKNNTest, ReturnsCorrectSizeGrayMask) {
    BackgroundSubtractKNN sub;
    auto fg = sub(make_bgr(48, 64));
    EXPECT_EQ(fg.mat().rows, 48);
    EXPECT_EQ(fg.mat().cols, 64);
    EXPECT_EQ(fg.mat().type(), CV_8UC1);
}

TEST(BackgroundSubtractKNNTest, PipelineOperatorSyntax) {
    BackgroundSubtractKNN sub;
    Image<Gray> fg = make_bgr(24, 32) | sub;
    EXPECT_EQ(fg.mat().rows, 24);
    EXPECT_EQ(fg.mat().cols, 32);
}

TEST(BackgroundSubtractKNNTest, FluentSettersReturnThis) {
    BackgroundSubtractKNN sub;
    auto& ref1 = sub.history(200);
    auto& ref2 = sub.threshold(500.0);
    auto& ref3 = sub.detect_shadows(true);
    EXPECT_EQ(&ref1, &sub);
    EXPECT_EQ(&ref2, &sub);
    EXPECT_EQ(&ref3, &sub);
}

TEST(BackgroundSubtractKNNTest, ModelUpdatesAcrossFrames) {
    BackgroundSubtractKNN sub;
    auto frame = make_bgr(24, 32);
    for (int i = 0; i < 5; ++i) sub(frame);
    auto different = Image<BGR>(cv::Mat(24, 32, CV_8UC3, cv::Scalar(0, 0, 0)));
    Image<Gray> fg = different | sub;
    EXPECT_GT(cv::countNonZero(fg.mat()), 0);
}
