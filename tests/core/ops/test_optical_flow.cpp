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

// ── DenseFarnebackFlow ────────────────────────────────────────────────────────

static Image<Gray> make_shifted_rect(int dx) {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(0));
    // Filled rectangle with internal checkerboard texture so Farneback has gradients to track
    for (int r = 60; r < 120; ++r)
        for (int c = 60 + dx; c < 120 + dx; ++c)
            m.at<uchar>(r, c) = static_cast<uchar>(((r / 5 + c / 5) % 2) ? 200 : 100);
    return Image<Gray>(m);
}

TEST(DenseFarnebackFlowTest, DefaultConstruction) {
    EXPECT_NO_THROW(DenseFarnebackFlow{});
}

TEST(DenseFarnebackFlowTest, FluentSettersReturnThis) {
    DenseFarnebackFlow op;
    EXPECT_EQ(&op.pyr_scale(0.5).levels(3).win_size(15).iterations(3).poly_n(5).poly_sigma(1.2), &op);
}

TEST(DenseFarnebackFlowTest, ReturnsFlowSameSize) {
    auto prev = make_shifted_rect(0);
    auto next = make_shifted_rect(10);
    auto flow = DenseFarnebackFlow{}(prev, next);
    EXPECT_EQ(flow.rows(), prev.rows());
    EXPECT_EQ(flow.cols(), prev.cols());
}

TEST(DenseFarnebackFlowTest, DetectsHorizontalTranslation) {
    auto prev = make_shifted_rect(0);
    auto next = make_shifted_rect(10);
    auto flow = DenseFarnebackFlow{}(prev, next);
    // Average flow inside the rectangle region should be approximately (10, 0)
    cv::Rect roi(65, 65, 50, 50);
    cv::Mat region = flow.mat()(roi);
    cv::Scalar mean_flow = cv::mean(region);
    EXPECT_NEAR(mean_flow[0], 10.0, 3.0);  // dx ≈ 10
    EXPECT_NEAR(mean_flow[1],  0.0, 3.0);  // dy ≈ 0
}

TEST(DenseFarnebackFlowTest, MismatchedSizesThrow) {
    Image<Gray> prev(cv::Mat(100, 100, CV_8UC1, cv::Scalar(0)));
    Image<Gray> next(cv::Mat(200, 200, CV_8UC1, cv::Scalar(0)));
    EXPECT_THROW(DenseFarnebackFlow{}(prev, next), std::invalid_argument);
}
