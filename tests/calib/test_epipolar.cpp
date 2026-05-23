// tests/calib/test_epipolar.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "improc/calib/pipeline.hpp"

using namespace improc::calib;

namespace {

cv::Mat make_K() {
    return (cv::Mat_<double>(3,3) << 600, 0, 320, 0, 600, 240, 0, 0, 1);
}

cv::Mat zero_dist() { return cv::Mat::zeros(1, 5, CV_64F); }

// 4×4 grid of 3D points on Z=0 plane, 50mm spacing (16 points)
std::vector<cv::Point3f> make_scene_pts() {
    std::vector<cv::Point3f> pts;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            pts.push_back({c * 50.f, r * 50.f, 0.f});
    return pts;
}

std::vector<cv::Point2f> project(const std::vector<cv::Point3f>& scene,
                                  const cv::Mat& K,
                                  const cv::Mat& rvec,
                                  const cv::Mat& tvec) {
    std::vector<cv::Point2f> pts;
    cv::projectPoints(scene, rvec, tvec, K, zero_dist(), pts);
    return pts;
}

// 3×4 projection matrix P = K * [R | t]
cv::Mat make_P(const cv::Mat& K, const cv::Mat& rvec, const cv::Mat& tvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    cv::Mat Rt;
    cv::hconcat(R, tvec, Rt);
    return K * Rt;
}

} // namespace

// ── FindFundamentalMat ────────────────────────────────────────────────────────

TEST(FindFundamentalMatTest, ThrowsOnSizeMismatch) {
    auto scene = make_scene_pts();
    auto K = make_K();
    auto rvec1 = cv::Mat(cv::Vec3d{0, 0, 0});
    auto tvec1 = cv::Mat(cv::Vec3d{0, 0, 600});
    auto tvec2 = cv::Mat(cv::Vec3d{-100, 0, 600});
    auto pts1 = project(scene, K, rvec1, tvec1);
    auto pts2 = project(scene, K, rvec1, tvec2);
    pts2.pop_back();
    EXPECT_THROW(FindFundamentalMat{}(pts1, pts2), std::invalid_argument);
}

TEST(FindFundamentalMatTest, ThrowsOnFewerThanEightPoints) {
    std::vector<cv::Point2f> pts(7, {1.f, 1.f});
    EXPECT_THROW(FindFundamentalMat{}(pts, pts), std::invalid_argument);
}

TEST(FindFundamentalMatTest, FIsThreeByThree) {
    auto scene = make_scene_pts();
    auto K = make_K();
    auto rvec1 = cv::Mat(cv::Vec3d{0, 0, 0});
    auto rvec2 = cv::Mat(cv::Vec3d{0, 0.1, 0}); // slight rotation breaks planar degeneracy
    auto tvec1 = cv::Mat(cv::Vec3d{0, 0, 600});
    auto tvec2 = cv::Mat(cv::Vec3d{-100, 0, 600});
    auto pts1 = project(scene, K, rvec1, tvec1);
    auto pts2 = project(scene, K, rvec2, tvec2);
    auto res = FindFundamentalMat{}(pts1, pts2);
    EXPECT_EQ(res.F.rows, 3);
    EXPECT_EQ(res.F.cols, 3);
    EXPECT_FALSE(res.mask.empty());
}

TEST(FindFundamentalMatTest, EpipolarConstraintHoldsForInliers) {
    auto scene = make_scene_pts();
    auto K = make_K();
    auto rvec1 = cv::Mat(cv::Vec3d{0, 0, 0});
    auto rvec2 = cv::Mat(cv::Vec3d{0, 0.1, 0}); // slight rotation breaks planar degeneracy
    auto tvec1 = cv::Mat(cv::Vec3d{0, 0, 600});
    auto tvec2 = cv::Mat(cv::Vec3d{-100, 0, 600});
    auto pts1 = project(scene, K, rvec1, tvec1);
    auto pts2 = project(scene, K, rvec2, tvec2);
    auto res = FindFundamentalMat{}(pts1, pts2);

    for (int i = 0; i < static_cast<int>(pts1.size()); ++i) {
        if (res.mask.at<uint8_t>(i) == 0) continue;
        cv::Mat x1 = (cv::Mat_<double>(3,1) << pts1[i].x, pts1[i].y, 1.0);
        cv::Mat x2 = (cv::Mat_<double>(1,3) << pts2[i].x, pts2[i].y, 1.0);
        cv::Mat val = x2 * res.F * x1;
        EXPECT_NEAR(val.at<double>(0), 0.0, 0.01)
            << "epipolar constraint violated at inlier " << i;
    }
}

TEST(FindFundamentalMatTest, FluentSettersReturnThis) {
    FindFundamentalMat op;
    EXPECT_EQ(&op.method(cv::FM_RANSAC), &op);
    EXPECT_EQ(&op.ransac_threshold(3.0), &op);
    EXPECT_EQ(&op.confidence(0.99), &op);
}

// ── FindEssentialMat ──────────────────────────────────────────────────────────

TEST(FindEssentialMatTest, ThrowsOnSizeMismatch) {
    auto scene = make_scene_pts();
    auto K = make_K();
    auto rvec = cv::Mat(cv::Vec3d{0, 0, 0});
    auto tvec1 = cv::Mat(cv::Vec3d{0, 0, 600});
    auto tvec2 = cv::Mat(cv::Vec3d{-100, 0, 600});
    auto pts1 = project(scene, K, rvec, tvec1);
    auto pts2 = project(scene, K, rvec, tvec2);
    pts2.pop_back();
    EXPECT_THROW(FindEssentialMat{}(pts1, pts2, K), std::invalid_argument);
}

TEST(FindEssentialMatTest, ThrowsOnFewerThanFivePoints) {
    std::vector<cv::Point2f> pts(4, {1.f, 1.f});
    EXPECT_THROW(FindEssentialMat{}(pts, pts, make_K()), std::invalid_argument);
}

TEST(FindEssentialMatTest, EIsThreeByThree) {
    auto scene = make_scene_pts();
    auto K = make_K();
    auto rvec = cv::Mat(cv::Vec3d{0, 0, 0});
    auto tvec1 = cv::Mat(cv::Vec3d{0, 0, 600});
    auto tvec2 = cv::Mat(cv::Vec3d{-100, 0, 600});
    auto pts1 = project(scene, K, rvec, tvec1);
    auto pts2 = project(scene, K, rvec, tvec2);
    auto res = FindEssentialMat{}(pts1, pts2, K);
    EXPECT_EQ(res.E.rows, 3);
    EXPECT_EQ(res.E.cols, 3);
    EXPECT_FALSE(res.mask.empty());
}

TEST(FindEssentialMatTest, RankTwoPropertyHolds) {
    auto scene = make_scene_pts();
    auto K = make_K();
    auto rvec = cv::Mat(cv::Vec3d{0, 0, 0});
    auto tvec1 = cv::Mat(cv::Vec3d{0, 0, 600});
    auto tvec2 = cv::Mat(cv::Vec3d{-100, 0, 600});
    auto pts1 = project(scene, K, rvec, tvec1);
    auto pts2 = project(scene, K, rvec, tvec2);
    auto res = FindEssentialMat{}(pts1, pts2, K);
    EXPECT_NEAR(cv::determinant(res.E), 0.0, 1e-6)
        << "det(E) should be ~0 (rank-2 property)";
}

TEST(FindEssentialMatTest, FluentSettersReturnThis) {
    FindEssentialMat op;
    EXPECT_EQ(&op.method(cv::RANSAC), &op);
    EXPECT_EQ(&op.threshold(1.0), &op);
    EXPECT_EQ(&op.confidence(0.99), &op);
}

// ── RecoverPose ───────────────────────────────────────────────────────────────

TEST(RecoverPoseTest, HasPositiveInliers) {
    auto scene = make_scene_pts();
    auto K = make_K();
    auto rvec = cv::Mat(cv::Vec3d{0, 0, 0});
    auto tvec1 = cv::Mat(cv::Vec3d{0, 0, 600});
    auto tvec2 = cv::Mat(cv::Vec3d{-100, 0, 600});
    auto pts1 = project(scene, K, rvec, tvec1);
    auto pts2 = project(scene, K, rvec, tvec2);
    auto eres = FindEssentialMat{}(pts1, pts2, K);
    auto pose = RecoverPose{}(eres.E, pts1, pts2, K);
    EXPECT_GT(pose.inliers, 0);
}

TEST(RecoverPoseTest, RIsOrthogonal) {
    auto scene = make_scene_pts();
    auto K = make_K();
    auto rvec = cv::Mat(cv::Vec3d{0, 0, 0});
    auto tvec1 = cv::Mat(cv::Vec3d{0, 0, 600});
    auto tvec2 = cv::Mat(cv::Vec3d{-100, 0, 600});
    auto pts1 = project(scene, K, rvec, tvec1);
    auto pts2 = project(scene, K, rvec, tvec2);
    auto eres = FindEssentialMat{}(pts1, pts2, K);
    auto pose = RecoverPose{}(eres.E, pts1, pts2, K);
    EXPECT_NEAR(cv::determinant(pose.R), 1.0, 1e-6)
        << "det(R) should be ~1.0 (proper rotation)";
}

TEST(RecoverPoseTest, TranslationIsUnitLength) {
    auto scene = make_scene_pts();
    auto K = make_K();
    auto rvec = cv::Mat(cv::Vec3d{0, 0, 0});
    auto tvec1 = cv::Mat(cv::Vec3d{0, 0, 600});
    auto tvec2 = cv::Mat(cv::Vec3d{-100, 0, 600});
    auto pts1 = project(scene, K, rvec, tvec1);
    auto pts2 = project(scene, K, rvec, tvec2);
    auto eres = FindEssentialMat{}(pts1, pts2, K);
    auto pose = RecoverPose{}(eres.E, pts1, pts2, K);
    double norm = cv::norm(pose.t);
    EXPECT_NEAR(norm, 1.0, 1e-6)
        << "|t|₂ should be 1.0 (unit translation)";
}

// ── TriangulatePoints ─────────────────────────────────────────────────────────

TEST(TriangulatePointsTest, OutputIsFourByN) {
    auto scene = make_scene_pts();
    auto K = make_K();
    auto rvec = cv::Mat(cv::Vec3d{0, 0, 0});
    auto tvec1 = cv::Mat(cv::Vec3d{0, 0, 600});
    auto tvec2 = cv::Mat(cv::Vec3d{-100, 0, 600});
    cv::Mat P1 = make_P(K, rvec, tvec1);
    cv::Mat P2 = make_P(K, rvec, tvec2);
    auto pts1 = project(scene, K, rvec, tvec1);
    auto pts2 = project(scene, K, rvec, tvec2);
    cv::Mat pts4d = TriangulatePoints{}(P1, P2, pts1, pts2);
    EXPECT_EQ(pts4d.rows, 4);
    EXPECT_EQ(pts4d.cols, static_cast<int>(scene.size()));
}

TEST(TriangulatePointsTest, TriangulatedPointsCloseToGroundTruth) {
    auto scene = make_scene_pts();
    auto K = make_K();
    auto rvec = cv::Mat(cv::Vec3d{0, 0, 0});
    auto tvec1 = cv::Mat(cv::Vec3d{0, 0, 600});
    auto tvec2 = cv::Mat(cv::Vec3d{-100, 0, 600});
    cv::Mat P1 = make_P(K, rvec, tvec1);
    cv::Mat P2 = make_P(K, rvec, tvec2);
    auto pts1 = project(scene, K, rvec, tvec1);
    auto pts2 = project(scene, K, rvec, tvec2);
    cv::Mat pts4d = TriangulatePoints{}(P1, P2, pts1, pts2);

    // cv::triangulatePoints returns points in world frame (homogeneous).
    // With R=I and scene on Z=0: X_world = (x, y, 0).
    for (int i = 0; i < pts4d.cols; ++i) {
        float w = pts4d.at<float>(3, i);
        ASSERT_GT(std::abs(w), 1e-6f) << "degenerate homogeneous point at " << i;
        float x_world = pts4d.at<float>(0, i) / w;
        float y_world = pts4d.at<float>(1, i) / w;
        float z_world = pts4d.at<float>(2, i) / w;

        EXPECT_NEAR(x_world, scene[i].x, scene[i].x * 0.01f + 1.f)
            << "x mismatch at point " << i;
        EXPECT_NEAR(y_world, scene[i].y, scene[i].y * 0.01f + 1.f)
            << "y mismatch at point " << i;
        EXPECT_NEAR(z_world, 0.f, 1.f)
            << "z mismatch at point " << i;
    }
}
