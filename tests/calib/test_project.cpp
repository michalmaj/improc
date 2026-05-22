// tests/calib/test_project.cpp
#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include "improc/calib/pipeline.hpp"

using namespace improc::calib;

namespace {
cv::Mat make_K() {
    return (cv::Mat_<double>(3,3) << 800,0,320, 0,800,240, 0,0,1);
}
cv::Mat zero_dist() { return cv::Mat::zeros(1, 5, CV_64F); }

// 6×4 inner corners, 30mm squares
std::vector<cv::Point3f> make_obj() {
    return make_chessboard_points({6, 4}, 30.f);
}

cv::Mat make_rvec() { return cv::Mat(cv::Vec3d{0.1, 0.05, 0.0}); }
cv::Mat make_tvec() { return cv::Mat(cv::Vec3d{0.0, 0.0, 500.0}); }
} // namespace

// ── ProjectPoints ─────────────────────────────────────────────────────────────

TEST(ProjectPointsTest, OutputCountMatchesInput) {
    auto obj = make_obj();
    auto pts = ProjectPoints{}(obj, make_rvec(), make_tvec(), make_K(), zero_dist());
    EXPECT_EQ(pts.size(), obj.size());
}

TEST(ProjectPointsTest, PointsWithinReasonableBounds) {
    auto pts = ProjectPoints{}(make_obj(), make_rvec(), make_tvec(), make_K(), zero_dist());
    for (const auto& p : pts) {
        EXPECT_GT(p.x, 0.f);
        EXPECT_LT(p.x, 640.f);
        EXPECT_GT(p.y, 0.f);
        EXPECT_LT(p.y, 480.f);
    }
}

// ── SolvePnP ─────────────────────────────────────────────────────────────────

TEST(SolvePnPTest, SucceedsOnSyntheticData) {
    auto obj = make_obj();
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(obj, make_rvec(), make_tvec(), make_K(), zero_dist(), img_pts);

    PnPResult r = SolvePnP{}(obj, img_pts, make_K(), zero_dist());
    EXPECT_TRUE(r.success);
    EXPECT_FALSE(r.rvec.empty());
    EXPECT_FALSE(r.tvec.empty());
}

TEST(SolvePnPTest, RoundTripReprojectionErrorLessThanOnePixel) {
    auto obj = make_obj();
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(obj, make_rvec(), make_tvec(), make_K(), zero_dist(), img_pts);

    PnPResult r = SolvePnP{}(obj, img_pts, make_K(), zero_dist());
    ASSERT_TRUE(r.success);

    auto reprojected = ProjectPoints{}(obj, r.rvec, r.tvec, make_K(), zero_dist());
    ASSERT_EQ(reprojected.size(), img_pts.size());

    double total_err = 0;
    for (size_t i = 0; i < img_pts.size(); ++i) {
        cv::Point2f d = img_pts[i] - reprojected[i];
        total_err += std::sqrt(d.x*d.x + d.y*d.y);
    }
    EXPECT_LT(total_err / img_pts.size(), 1.0);
}

TEST(SolvePnPTest, ThrowsOnFewerThanFourPoints) {
    std::vector<cv::Point3f> obj3 = {{0,0,0},{1,0,0},{0,1,0}};
    std::vector<cv::Point2f> img3 = {{100,100},{200,100},{100,200}};
    EXPECT_THROW(SolvePnP{}(obj3, img3, make_K(), zero_dist()), std::invalid_argument);
}

TEST(SolvePnPTest, ThrowsOnMismatchedSizes) {
    auto obj = make_obj();
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(obj, make_rvec(), make_tvec(), make_K(), zero_dist(), img_pts);
    img_pts.pop_back();
    EXPECT_THROW(SolvePnP{}(obj, img_pts, make_K(), zero_dist()), std::invalid_argument);
}

TEST(SolvePnPTest, FluentMethodReturnsThis) {
    SolvePnP op;
    EXPECT_EQ(&op.method(cv::SOLVEPNP_ITERATIVE), &op);
}

// ── SolvePnPRansac ────────────────────────────────────────────────────────────

TEST(SolvePnPRansacTest, SucceedsOnCleanData) {
    auto obj = make_obj();
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(obj, make_rvec(), make_tvec(), make_K(), zero_dist(), img_pts);

    PnPRansacResult r = SolvePnPRansac{}(obj, img_pts, make_K(), zero_dist());
    EXPECT_TRUE(r.success);
    EXPECT_FALSE(r.inliers.empty());
}

TEST(SolvePnPRansacTest, RobustToTwentyPercentOutliers) {
    auto obj = make_obj();
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(obj, make_rvec(), make_tvec(), make_K(), zero_dist(), img_pts);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise(-200.f, 200.f);
    int n_outliers = static_cast<int>(img_pts.size() * 0.2);
    for (int i = 0; i < n_outliers; ++i) {
        img_pts[i].x += noise(rng);
        img_pts[i].y += noise(rng);
    }

    PnPRansacResult r = SolvePnPRansac{}(obj, img_pts, make_K(), zero_dist());
    EXPECT_TRUE(r.success);
    EXPECT_FALSE(r.inliers.empty());
    EXPECT_GT(r.inliers.rows, static_cast<int>(obj.size()) * 70 / 100);
}

TEST(SolvePnPRansacTest, FluentSettersReturnThis) {
    SolvePnPRansac op;
    EXPECT_EQ(&op.method(cv::SOLVEPNP_ITERATIVE), &op);
    EXPECT_EQ(&op.confidence(0.99), &op);
    EXPECT_EQ(&op.reprojection_error(8.f), &op);
    EXPECT_EQ(&op.iterations(100), &op);
}
