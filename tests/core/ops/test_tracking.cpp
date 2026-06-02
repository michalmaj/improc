// tests/core/ops/test_tracking.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

static Image<Gray> make_prob_map(cv::Rect hot_rect, int rows = 200, int cols = 200) {
    cv::Mat m(rows, cols, CV_8UC1, cv::Scalar(0));
    m(hot_rect).setTo(cv::Scalar(255));
    return Image<Gray>(m);
}

// ── CamShift ──────────────────────────────────────────────────────────────────

TEST(CamShiftTest, DefaultConstruction) {
    EXPECT_NO_THROW(CamShift{});
}

TEST(CamShiftTest, FluentSettersReturnThis) {
    CamShift op;
    EXPECT_EQ(&op.epsilon(0.5).max_iter(20), &op);
}

TEST(CamShiftTest, ConvergesOnBrightRegion) {
    cv::Rect hot(80, 80, 40, 40);
    auto back_proj = make_prob_map(hot);
    cv::Rect window(60, 60, 50, 50);
    auto result = CamShift{}(back_proj, window);
    // Tracked object center should be near center of hot_rect (100, 100)
    EXPECT_NEAR(result.object.center.x, 100.f, 10.f);
    EXPECT_NEAR(result.object.center.y, 100.f, 10.f);
}

TEST(CamShiftTest, OutOfBoundsWindowThrows) {
    auto back_proj = make_prob_map(cv::Rect(80, 80, 40, 40));
    cv::Rect window(-10, -10, 50, 50);
    EXPECT_THROW(CamShift{}(back_proj, window), improc::ParameterError);
}

// ── MeanShift ─────────────────────────────────────────────────────────────────

TEST(MeanShiftTest, DefaultConstruction) {
    EXPECT_NO_THROW(MeanShift{});
}

TEST(MeanShiftTest, FluentSettersReturnThis) {
    MeanShift op;
    EXPECT_EQ(&op.epsilon(0.5).max_iter(20), &op);
}

TEST(MeanShiftTest, ConvergesOnBrightRegion) {
    cv::Rect hot(80, 80, 40, 40);
    auto back_proj = make_prob_map(hot);
    // Offset window so meanShift must move to converge
    cv::Rect window(60, 60, 50, 50);
    int iters = MeanShift{}(back_proj, window);
    EXPECT_GT(iters, 0);
    EXPECT_NEAR(window.x + window.width  / 2, 100, 10);
    EXPECT_NEAR(window.y + window.height / 2, 100, 10);
}

TEST(MeanShiftTest, OutOfBoundsWindowThrows) {
    auto back_proj = make_prob_map(cv::Rect(80, 80, 40, 40));
    cv::Rect window(250, 250, 50, 50);
    EXPECT_THROW(MeanShift{}(back_proj, window), improc::ParameterError);
}
