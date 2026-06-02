// tests/calib/test_fisheye.cpp
#include <gtest/gtest.h>
#include <vector>
#include <opencv2/calib3d.hpp>
#include "improc/calib/pipeline.hpp"

using namespace improc::calib;
using improc::core::BGR;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void make_fisheye_calib_data(
    std::vector<std::vector<cv::Point3f>>& obj_pts,
    std::vector<std::vector<cv::Point2f>>& img_pts,
    cv::Mat& true_K, cv::Mat& true_D,
    cv::Size& image_size)
{
    constexpr int W = 5, H = 5;
    constexpr float S = 30.0f;
    std::vector<cv::Point3f> board;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            board.push_back({c * S, r * S, 0.0f});

    image_size = {640, 480};
    true_K = (cv::Mat_<double>(3,3) << 300,0,320, 0,300,240, 0,0,1);
    true_D = (cv::Mat_<double>(4,1) << 0.01, 0.005, 0.001, 0.001);

    for (int v = 0; v < 5; ++v) {
        cv::Mat rvec = (cv::Mat_<double>(3,1) << 0.1*v, 0.05*v, 0.0);
        cv::Mat tvec = (cv::Mat_<double>(3,1) << -60 + 30*v, -30, 500);
        std::vector<cv::Point2f> proj;
        cv::fisheye::projectPoints(board, proj, rvec, tvec, true_K, true_D);
        obj_pts.push_back(board);
        img_pts.push_back(proj);
    }
}

static void make_fisheye_stereo_data(
    std::vector<std::vector<cv::Point3f>>& obj_pts,
    std::vector<std::vector<cv::Point2f>>& left_pts,
    std::vector<std::vector<cv::Point2f>>& right_pts,
    cv::Mat& K1, cv::Mat& D1,
    cv::Mat& K2, cv::Mat& D2,
    cv::Size& image_size)
{
    constexpr int W = 5, H = 5;
    constexpr float S = 30.0f;
    std::vector<cv::Point3f> board;
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c)
            board.push_back({c * S, r * S, 0.0f});

    image_size = {640, 480};
    K1 = (cv::Mat_<double>(3,3) << 300,0,320, 0,300,240, 0,0,1);
    D1 = (cv::Mat_<double>(4,1) << 0.01, 0.005, 0.001, 0.001);
    K2 = (cv::Mat_<double>(3,3) << 300,0,320, 0,300,240, 0,0,1);
    D2 = (cv::Mat_<double>(4,1) << 0.01, 0.005, 0.001, 0.001);

    // Baseline: right camera is shifted 60 mm in X
    const double baseline = 60.0;
    for (int v = 0; v < 5; ++v) {
        cv::Mat rvec  = (cv::Mat_<double>(3,1) << 0.1*v, 0.05*v, 0.0);
        cv::Mat tvec1 = (cv::Mat_<double>(3,1) << -60 + 30*v, -30, 500);
        cv::Mat tvec2 = (cv::Mat_<double>(3,1) << -60 + 30*v + baseline, -30, 500);

        std::vector<cv::Point2f> lp, rp;
        cv::fisheye::projectPoints(board, lp, rvec, tvec1, K1, D1);
        cv::fisheye::projectPoints(board, rp, rvec, tvec2, K2, D2);
        obj_pts.push_back(board);
        left_pts.push_back(lp);
        right_pts.push_back(rp);
    }
}

// ── FisheyeCalibrate ──────────────────────────────────────────────────────────

TEST(FisheyeCalibrateTest, ReturnsCalibrationResult) {
    std::vector<std::vector<cv::Point3f>> obj_pts;
    std::vector<std::vector<cv::Point2f>> img_pts;
    cv::Mat K, D;
    cv::Size sz;
    make_fisheye_calib_data(obj_pts, img_pts, K, D, sz);

    CalibrationResult result = FisheyeCalibrate{}(obj_pts, img_pts, sz);
    EXPECT_FALSE(result.camera_matrix.empty());
    EXPECT_FALSE(result.dist_coeffs.empty());
    EXPECT_GT(result.rvecs.size(), 0u);
    EXPECT_LT(result.rms, 5.0);
}

TEST(FisheyeCalibrateTest, FluentSettersReturnThis) {
    FisheyeCalibrate op;
    EXPECT_EQ(&op.flags(cv::fisheye::CALIB_FIX_SKEW), &op);
    EXPECT_EQ(&op.criteria(cv::TermCriteria{}), &op);
}

// ── FisheyeUndistort ──────────────────────────────────────────────────────────

TEST(FisheyeUndistortTest, OutputSameSize) {
    cv::Mat K = (cv::Mat_<double>(3,3) << 300,0,320, 0,300,240, 0,0,1);
    cv::Mat D = (cv::Mat_<double>(4,1) << 0.01, 0.005, 0.001, 0.001);
    cv::Mat m(480, 640, CV_8UC3, cv::Scalar(128, 64, 32));
    Image<BGR> img(m);
    Image<BGR> result = FisheyeUndistort{}.K(K).dist(D)(img);
    EXPECT_EQ(result.rows(), 480);
    EXPECT_EQ(result.cols(), 640);
}

TEST(FisheyeUndistortTest, FluentSettersReturnThis) {
    FisheyeUndistort op;
    cv::Mat m;
    EXPECT_EQ(&op.K(m), &op);
    EXPECT_EQ(&op.dist(m), &op);
    EXPECT_EQ(&op.new_K(m), &op);
}

// ── FisheyeUndistortPoints ────────────────────────────────────────────────────

TEST(FisheyeUndistortPointsTest, ReturnsCorrectCount) {
    cv::Mat K = (cv::Mat_<double>(3,3) << 300,0,320, 0,300,240, 0,0,1);
    cv::Mat D = (cv::Mat_<double>(4,1) << 0.01, 0.005, 0.001, 0.001);
    std::vector<cv::Point2f> pts = {{100.f, 100.f}, {200.f, 200.f}, {320.f, 240.f}};
    auto result = FisheyeUndistortPoints{}.K(K).dist(D)(pts);
    EXPECT_EQ(result.size(), 3u);
}

TEST(FisheyeUndistortPointsTest, FluentSettersReturnThis) {
    FisheyeUndistortPoints op;
    cv::Mat m;
    EXPECT_EQ(&op.K(m), &op);
    EXPECT_EQ(&op.dist(m), &op);
    EXPECT_EQ(&op.R(m), &op);
    EXPECT_EQ(&op.P(m), &op);
}

// ── FisheyeInitRectifyMap ─────────────────────────────────────────────────────

TEST(FisheyeInitRectifyMapTest, ProducesNonEmptyMaps) {
    cv::Mat K = (cv::Mat_<double>(3,3) << 300,0,320, 0,300,240, 0,0,1);
    cv::Mat D = (cv::Mat_<double>(4,1) << 0.01, 0.005, 0.001, 0.001);
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
    UndistortMapResult maps = FisheyeInitRectifyMap{}.K(K).dist(D).R(R).new_K(K)(cv::Size{640,480});
    EXPECT_FALSE(maps.map1.empty());
    EXPECT_FALSE(maps.map2.empty());
    EXPECT_EQ(maps.map1.size(), maps.map2.size());
}

TEST(FisheyeInitRectifyMapTest, MapsHaveCorrectSize) {
    cv::Mat K = (cv::Mat_<double>(3,3) << 300,0,320, 0,300,240, 0,0,1);
    cv::Mat D = (cv::Mat_<double>(4,1) << 0.01, 0.005, 0.001, 0.001);
    cv::Size sz{640, 480};
    UndistortMapResult maps = FisheyeInitRectifyMap{}.K(K).dist(D)(sz);
    EXPECT_EQ(maps.map1.rows, sz.height);
    EXPECT_EQ(maps.map1.cols, sz.width);
}

TEST(FisheyeInitRectifyMapTest, FluentSettersReturnThis) {
    FisheyeInitRectifyMap op;
    cv::Mat m;
    EXPECT_EQ(&op.K(m), &op);
    EXPECT_EQ(&op.dist(m), &op);
    EXPECT_EQ(&op.R(m), &op);
    EXPECT_EQ(&op.new_K(m), &op);
}

// ── FisheyeStereoCalibrate ────────────────────────────────────────────────────

TEST(FisheyeStereoCalibrateTest, ReturnsPopulatedResult) {
    std::vector<std::vector<cv::Point3f>> obj_pts;
    std::vector<std::vector<cv::Point2f>> left_pts, right_pts;
    cv::Mat K1, D1, K2, D2;
    cv::Size sz;
    make_fisheye_stereo_data(obj_pts, left_pts, right_pts, K1, D1, K2, D2, sz);

    StereoCalibrationResult result = FisheyeStereoCalibrate{}
        .K1(K1).dist1(D1).K2(K2).dist2(D2)
        .flags(cv::fisheye::CALIB_FIX_INTRINSIC)
        (obj_pts, left_pts, right_pts, sz);

    EXPECT_FALSE(result.K1.empty());
    EXPECT_FALSE(result.K2.empty());
    EXPECT_FALSE(result.R.empty());
    EXPECT_FALSE(result.T.empty());
    EXPECT_LT(result.rms, 5.0);
}

TEST(FisheyeStereoCalibrateTest, FluentSettersReturnThis) {
    FisheyeStereoCalibrate op;
    cv::Mat m;
    EXPECT_EQ(&op.K1(m), &op);
    EXPECT_EQ(&op.dist1(m), &op);
    EXPECT_EQ(&op.K2(m), &op);
    EXPECT_EQ(&op.dist2(m), &op);
    EXPECT_EQ(&op.flags(0), &op);
}

// ── FisheyeStereoRectify ──────────────────────────────────────────────────────

TEST(FisheyeStereoRectifyTest, ProducesCorrectShapes) {
    std::vector<std::vector<cv::Point3f>> obj_pts;
    std::vector<std::vector<cv::Point2f>> left_pts, right_pts;
    cv::Mat K1, D1, K2, D2;
    cv::Size sz;
    make_fisheye_stereo_data(obj_pts, left_pts, right_pts, K1, D1, K2, D2, sz);

    StereoCalibrationResult cal = FisheyeStereoCalibrate{}
        .K1(K1).dist1(D1).K2(K2).dist2(D2)
        .flags(cv::fisheye::CALIB_FIX_INTRINSIC)
        (obj_pts, left_pts, right_pts, sz);

    StereoRectifyResult rect = FisheyeStereoRectify{}
        .image_size(sz)
        (cal.K1, cal.dist1, cal.K2, cal.dist2, cal.R, cal.T);

    EXPECT_EQ(rect.R1.rows, 3);  EXPECT_EQ(rect.R1.cols, 3);
    EXPECT_EQ(rect.R2.rows, 3);  EXPECT_EQ(rect.R2.cols, 3);
    EXPECT_EQ(rect.P1.rows, 3);  EXPECT_EQ(rect.P1.cols, 4);
    EXPECT_EQ(rect.P2.rows, 3);  EXPECT_EQ(rect.P2.cols, 4);
    EXPECT_EQ(rect.Q.rows, 4);   EXPECT_EQ(rect.Q.cols, 4);
}

TEST(FisheyeStereoRectifyTest, FluentSettersReturnThis) {
    FisheyeStereoRectify op;
    EXPECT_EQ(&op.image_size({640,480}), &op);
    EXPECT_EQ(&op.flags(cv::CALIB_ZERO_DISPARITY), &op);
    EXPECT_EQ(&op.balance(0.5), &op);
    EXPECT_EQ(&op.fov_scale(1.0), &op);
}
