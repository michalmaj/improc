// tests/core/ops/test_optical_flow.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

static Image<Gray> make_circle_gray(cv::Point center) {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(0));
    cv::circle(m, center, 20, cv::Scalar(255), -1);
    return Image<Gray>(m);
}

// ── SparseLKFlow ──────────────────────────────────────────────────────────────

TEST(SparseLKFlowTest, DefaultConstruction) {
    EXPECT_NO_THROW(SparseLKFlow{});
}

TEST(SparseLKFlowTest, FluentSettersReturnThis) {
    SparseLKFlow op;
    EXPECT_EQ(&op.win_size({31, 31}).max_level(2).max_iter(20).epsilon(0.05), &op);
}

TEST(SparseLKFlowTest, TracksKnownTranslation) {
    auto prev = make_circle_gray({80, 100});
    auto next = make_circle_gray({90, 100});  // circle shifted 10px right (x+10)
    std::vector<cv::Point2f> pts = {{80.f, 100.f}};
    auto r = SparseLKFlow{}.win_size({31, 31})(prev, next, pts);
    ASSERT_EQ(r.status.size(), 1u);
    ASSERT_EQ(r.points.size(), 1u);
    EXPECT_EQ(r.status[0], 1);
    EXPECT_NEAR(r.points[0].x, 90.f, 3.f);
    EXPECT_NEAR(r.points[0].y, 100.f, 3.f);
}

TEST(SparseLKFlowTest, EmptyPointsThrows) {
    auto prev = make_circle_gray({80, 100});
    auto next = make_circle_gray({90, 100});
    EXPECT_THROW(SparseLKFlow{}(prev, next, {}), std::invalid_argument);
}

TEST(SparseLKFlowTest, MismatchedSizesThrow) {
    Image<Gray> prev(cv::Mat(100, 100, CV_8UC1, cv::Scalar(0)));
    Image<Gray> next(cv::Mat(200, 200, CV_8UC1, cv::Scalar(0)));
    std::vector<cv::Point2f> pts = {{50.f, 50.f}};
    EXPECT_THROW(SparseLKFlow{}(prev, next, pts), std::invalid_argument);
}
