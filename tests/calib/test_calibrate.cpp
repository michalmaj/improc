#include <gtest/gtest.h>
#include <cmath>
#include "improc/calib/pipeline.hpp"

using namespace improc::calib;

namespace {
// Projects 3D points to 2D using a known camera
std::vector<cv::Point2f> project(const std::vector<cv::Point3f>& obj,
                                  const cv::Mat& K, const cv::Mat& dist,
                                  const cv::Mat& rvec, const cv::Mat& tvec) {
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(obj, rvec, tvec, K, dist, img_pts);
    return img_pts;
}

// Builds synthetic calibration data using a known pinhole camera
struct CalibData {
    std::vector<std::vector<cv::Point3f>> obj_pts;
    std::vector<std::vector<cv::Point2f>> img_pts;
    cv::Size image_size;
    cv::Mat  K_gt;
};

CalibData make_calib_data() {
    const cv::Size board{9, 6};
    const float sq = 30.f;
    const cv::Size img_size{640, 480};
    const double fx = 600, fy = 600, cx = 320, cy = 240;

    cv::Mat K_gt = (cv::Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);
    cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);

    auto obj = make_chessboard_points(board, sq);

    // Different board orientations
    std::vector<cv::Vec3d> rvecs_gt = {
        {0.0, 0.0, 0.0}, {0.15, 0.0, 0.0}, {-0.15, 0.0, 0.0},
        {0.0, 0.15, 0.0}, {0.0,-0.15, 0.0}, { 0.10, 0.10, 0.0},
        {-0.10, 0.10, 0.0}, {0.10,-0.10, 0.0}, {-0.10,-0.10, 0.0},
        {0.20, 0.0, 0.0}, {-0.20, 0.0, 0.0}, {0.0, 0.20, 0.0},
    };
    cv::Vec3d t_gt{-120, -90, 600};

    CalibData data;
    data.image_size = img_size;
    data.K_gt       = K_gt;

    for (const auto& rv : rvecs_gt) {
        cv::Mat rvec = cv::Mat(rv), tvec = cv::Mat(t_gt);
        auto img_pts = project(obj, K_gt, dist, rvec, tvec);
        bool in_bounds = std::all_of(img_pts.begin(), img_pts.end(),
            [&](const cv::Point2f& p) {
                return p.x >= 0 && p.x < img_size.width &&
                       p.y >= 0 && p.y < img_size.height;
            });
        if (in_bounds) {
            data.obj_pts.push_back(obj);
            data.img_pts.push_back(img_pts);
        }
    }
    return data;
}
} // namespace

// ── make_chessboard_points ───────────────────────────────────────────────────

TEST(MakeChessboardPointsTest, CountIsWidthTimesHeight) {
    auto pts = make_chessboard_points({9, 6}, 25.f);
    EXPECT_EQ(pts.size(), static_cast<size_t>(9 * 6));
}

TEST(MakeChessboardPointsTest, AllZsAreZero) {
    for (const auto& p : make_chessboard_points({9, 6}, 25.f))
        EXPECT_FLOAT_EQ(p.z, 0.f);
}

TEST(MakeChessboardPointsTest, SpacingEqualsSquareSize) {
    const float sq = 25.f;
    auto pts = make_chessboard_points({9, 6}, sq);
    EXPECT_FLOAT_EQ(pts[1].x - pts[0].x, sq);
    EXPECT_FLOAT_EQ(pts[9].y - pts[0].y, sq);
}

TEST(MakeChessboardPointsTest, FirstPointIsOrigin) {
    auto pts = make_chessboard_points({9, 6}, 25.f);
    EXPECT_FLOAT_EQ(pts[0].x, 0.f);
    EXPECT_FLOAT_EQ(pts[0].y, 0.f);
    EXPECT_FLOAT_EQ(pts[0].z, 0.f);
}

TEST(MakeChessboardPointsTest, ThrowsOnZeroWidth) {
    EXPECT_THROW(make_chessboard_points({0, 6}, 25.f), std::invalid_argument);
}

TEST(MakeChessboardPointsTest, ThrowsOnNegativeDimension) {
    EXPECT_THROW(make_chessboard_points({9, -1}, 25.f), std::invalid_argument);
}

TEST(MakeChessboardPointsTest, ThrowsOnNonPositiveSquareSize) {
    EXPECT_THROW(make_chessboard_points({9, 6}, 0.f), std::invalid_argument);
}

// ── CalibrateCamera ──────────────────────────────────────────────────────────

TEST(CalibrateCameraTest, ThrowsOnMismatchedVectorSizes) {
    auto data = make_calib_data();
    auto obj = data.obj_pts;
    auto img = data.img_pts;
    img.pop_back();
    EXPECT_THROW(CalibrateCamera{}(obj, img, data.image_size), std::invalid_argument);
}

TEST(CalibrateCameraTest, ThrowsOnFewerThanThreeViews) {
    auto data = make_calib_data();
    std::vector<std::vector<cv::Point3f>> obj2 = {data.obj_pts[0], data.obj_pts[1]};
    std::vector<std::vector<cv::Point2f>> img2 = {data.img_pts[0], data.img_pts[1]};
    EXPECT_THROW(CalibrateCamera{}(obj2, img2, data.image_size), std::invalid_argument);
}

TEST(CalibrateCameraTest, ResultStructIsPopulated) {
    auto data = make_calib_data();
    auto cal = CalibrateCamera{}(data.obj_pts, data.img_pts, data.image_size);
    EXPECT_EQ(cal.camera_matrix.rows, 3);
    EXPECT_EQ(cal.camera_matrix.cols, 3);
    EXPECT_FALSE(cal.dist_coeffs.empty());
    EXPECT_EQ(cal.rvecs.size(), data.obj_pts.size());
    EXPECT_EQ(cal.tvecs.size(), data.obj_pts.size());
    EXPECT_GE(cal.rms, 0.0);
}

TEST(CalibrateCameraTest, RmsLowOnSyntheticData) {
    auto data = make_calib_data();
    auto cal = CalibrateCamera{}(data.obj_pts, data.img_pts, data.image_size);
    EXPECT_LT(cal.rms, 1.0) << "RMS=" << cal.rms;
}

TEST(CalibrateCameraTest, RecoveredFocalLengthCloseToGroundTruth) {
    auto data  = make_calib_data();
    auto cal   = CalibrateCamera{}(data.obj_pts, data.img_pts, data.image_size);
    double fx_gt  = data.K_gt.at<double>(0, 0);
    double fx_est = cal.camera_matrix.at<double>(0, 0);
    EXPECT_NEAR(fx_est, fx_gt, fx_gt * 0.05)
        << "fx_gt=" << fx_gt << " fx_est=" << fx_est;
}
