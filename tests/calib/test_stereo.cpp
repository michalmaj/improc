// tests/calib/test_stereo.cpp
#include <gtest/gtest.h>
#include <cmath>
#include "improc/calib/pipeline.hpp"

using namespace improc::calib;

namespace {

struct StereoData {
    std::vector<std::vector<cv::Point3f>> obj_pts;
    std::vector<std::vector<cv::Point2f>> img_pts1;
    std::vector<std::vector<cv::Point2f>> img_pts2;
    cv::Size image_size;
    cv::Mat  K_gt;
    double   T_gt_x;
};

StereoData make_stereo_data() {
    const cv::Size board{9, 6};
    const float sq = 30.f;
    const cv::Size img_size{640, 480};
    const double fx = 600, fy = 600, cx = 320, cy = 240;

    cv::Mat K = (cv::Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);
    cv::Mat dist = cv::Mat::zeros(1, 5, CV_64F);
    cv::Vec3d T_gt{60.0, 0.0, 0.0};
    cv::Vec3d t_gt{-120.0, -90.0, 600.0};

    std::vector<cv::Vec3d> rvecs = {
        {0.0, 0.0, 0.0}, {0.15, 0.0, 0.0}, {-0.15, 0.0, 0.0},
        {0.0, 0.15, 0.0}, {0.0,-0.15, 0.0}, { 0.10, 0.10, 0.0},
        {-0.10, 0.10, 0.0}, {0.10,-0.10, 0.0}, {-0.10,-0.10, 0.0},
        {0.20, 0.0, 0.0}, {-0.20, 0.0, 0.0}, {0.0, 0.20, 0.0},
    };

    auto obj = make_chessboard_points(board, sq);

    StereoData data;
    data.image_size = img_size;
    data.K_gt = K;
    data.T_gt_x = T_gt[0];

    for (const auto& rv : rvecs) {
        cv::Mat rvec = cv::Mat(rv);
        cv::Mat tvec1 = cv::Mat(t_gt);
        cv::Mat tvec2 = cv::Mat(t_gt + T_gt);

        std::vector<cv::Point2f> pts1, pts2;
        cv::projectPoints(obj, rvec, tvec1, K, dist, pts1);
        cv::projectPoints(obj, rvec, tvec2, K, dist, pts2);

        auto in_bounds = [&](const std::vector<cv::Point2f>& pts) {
            return std::all_of(pts.begin(), pts.end(), [&](const cv::Point2f& p) {
                return p.x >= 0 && p.x < img_size.width &&
                       p.y >= 0 && p.y < img_size.height;
            });
        };
        if (in_bounds(pts1) && in_bounds(pts2)) {
            data.obj_pts.push_back(obj);
            data.img_pts1.push_back(pts1);
            data.img_pts2.push_back(pts2);
        }
    }
    return data;
}

// Returns {left_gray, right_gray} where right is left shifted by shift_px in X.
std::pair<Image<Gray>, Image<Gray>> make_stereo_pair(cv::Size board_size,
                                                      int square_px,
                                                      int shift_px) {
    int border = square_px;
    int cols = (board_size.width + 1) * square_px + 2 * border;
    int rows = (board_size.height + 1) * square_px + 2 * border;
    cv::Mat mat(rows, cols, CV_8UC1, cv::Scalar(255));
    for (int r = 0; r <= board_size.height; ++r)
        for (int c = 0; c <= board_size.width; ++c)
            if ((r + c) % 2 == 0) {
                cv::Rect rect(border + c * square_px,
                              border + r * square_px,
                              square_px, square_px);
                mat(rect).setTo(0);
            }
    Image<Gray> left(mat);
    cv::Mat shift_mat = (cv::Mat_<double>(2, 3) << 1, 0, -shift_px, 0, 1, 0);
    cv::Mat right_mat;
    cv::warpAffine(mat, right_mat, shift_mat, mat.size());
    return {left, Image<Gray>(right_mat)};
}

} // namespace

// ── StereoCalibrate ──────────────────────────────────────────────────────────

TEST(StereoCalibrateTest, ThrowsOnMismatchedVectorSizes) {
    auto d = make_stereo_data();
    auto obj = d.obj_pts;
    auto pts2 = d.img_pts2;
    pts2.pop_back();
    EXPECT_THROW(StereoCalibrate{}(obj, d.img_pts1, pts2, d.image_size),
                 std::invalid_argument);
}

TEST(StereoCalibrateTest, ThrowsOnFewerThanThreeViews) {
    auto d = make_stereo_data();
    std::vector<std::vector<cv::Point3f>> obj2 = {d.obj_pts[0], d.obj_pts[1]};
    std::vector<std::vector<cv::Point2f>> p1 = {d.img_pts1[0], d.img_pts1[1]};
    std::vector<std::vector<cv::Point2f>> p2 = {d.img_pts2[0], d.img_pts2[1]};
    EXPECT_THROW(StereoCalibrate{}(obj2, p1, p2, d.image_size), std::invalid_argument);
}

TEST(StereoCalibrateTest, ThrowsOnZeroImageSize) {
    auto d = make_stereo_data();
    EXPECT_THROW(StereoCalibrate{}(d.obj_pts, d.img_pts1, d.img_pts2, {0,0}),
                 std::invalid_argument);
}

TEST(StereoCalibrateTest, RmsLowOnSyntheticData) {
    auto d = make_stereo_data();
    auto cal = StereoCalibrate{}(d.obj_pts, d.img_pts1, d.img_pts2, d.image_size);
    EXPECT_LT(cal.rms, 1.0) << "rms=" << cal.rms;
}

TEST(StereoCalibrateTest, ResultMatricesArePopulated) {
    auto d = make_stereo_data();
    auto cal = StereoCalibrate{}(d.obj_pts, d.img_pts1, d.img_pts2, d.image_size);
    EXPECT_EQ(cal.K1.rows, 3);  EXPECT_EQ(cal.K1.cols, 3);
    EXPECT_EQ(cal.K2.rows, 3);  EXPECT_EQ(cal.K2.cols, 3);
    EXPECT_EQ(cal.R.rows, 3);   EXPECT_EQ(cal.R.cols, 3);
    EXPECT_EQ(cal.T.rows, 3);   EXPECT_EQ(cal.T.cols, 1);
    EXPECT_EQ(cal.E.rows, 3);   EXPECT_EQ(cal.E.cols, 3);
    EXPECT_EQ(cal.F.rows, 3);   EXPECT_EQ(cal.F.cols, 3);
}

TEST(StereoCalibrateTest, RotationIsCloseToIdentity) {
    auto d = make_stereo_data();
    auto cal = StereoCalibrate{}(d.obj_pts, d.img_pts1, d.img_pts2, d.image_size);
    double err = cv::norm(cal.R, cv::Mat::eye(3, 3, CV_64F));
    EXPECT_LT(err, 0.05) << "R deviation from I: " << err;
}

TEST(StereoCalibrateTest, BaselineCloseToGroundTruth) {
    auto d = make_stereo_data();
    auto cal = StereoCalibrate{}(d.obj_pts, d.img_pts1, d.img_pts2, d.image_size);
    double t0 = cal.T.at<double>(0);
    EXPECT_NEAR(t0, d.T_gt_x, std::abs(d.T_gt_x) * 0.05)
        << "T[0]=" << t0 << " expected~" << d.T_gt_x;
}

// ── StereoRectify ─────────────────────────────────────────────────────────────

TEST(StereoRectifyTest, ThrowsOnEmptyMatrices) {
    EXPECT_THROW(
        StereoRectify{}(cv::Mat{}, cv::Mat{}, cv::Mat{}, cv::Mat{},
                        cv::Mat{}, cv::Mat{}, {640,480}),
        std::invalid_argument);
}

TEST(StereoRectifyTest, OutputShapesAreCorrect) {
    auto d = make_stereo_data();
    auto cal = StereoCalibrate{}(d.obj_pts, d.img_pts1, d.img_pts2, d.image_size);
    auto rect = StereoRectify{}(cal.K1, cal.dist1, cal.K2, cal.dist2,
                                cal.R, cal.T, d.image_size);
    EXPECT_EQ(rect.R1.rows, 3);  EXPECT_EQ(rect.R1.cols, 3);
    EXPECT_EQ(rect.R2.rows, 3);  EXPECT_EQ(rect.R2.cols, 3);
    EXPECT_EQ(rect.P1.rows, 3);  EXPECT_EQ(rect.P1.cols, 4);
    EXPECT_EQ(rect.P2.rows, 3);  EXPECT_EQ(rect.P2.cols, 4);
    EXPECT_EQ(rect.Q.rows, 4);   EXPECT_EQ(rect.Q.cols, 4);
}

TEST(StereoRectifyTest, ValidROIsAreNonEmpty) {
    auto d = make_stereo_data();
    auto cal = StereoCalibrate{}(d.obj_pts, d.img_pts1, d.img_pts2, d.image_size);
    auto rect = StereoRectify{}(cal.K1, cal.dist1, cal.K2, cal.dist2,
                                cal.R, cal.T, d.image_size);
    EXPECT_GT(rect.validROI1.area(), 0);
    EXPECT_GT(rect.validROI2.area(), 0);
}

// ── StereoBM ──────────────────────────────────────────────────────────────────

TEST(StereoBMTest, ThrowsOnSizeMismatch) {
    auto [left, right] = make_stereo_pair({9,6}, 20, 16);
    cv::Mat smaller(left.mat().rows / 2, left.mat().cols / 2, CV_8UC1);
    EXPECT_THROW(StereoBM{}(left, Image<Gray>(smaller)), std::invalid_argument);
}

TEST(StereoBMTest, OutputIsCV16SAndMatchesInputSize) {
    auto [left, right] = make_stereo_pair({9,6}, 20, 16);
    cv::Mat disp = StereoBM{}(left, right);
    EXPECT_EQ(disp.type(), CV_16S);
    EXPECT_EQ(disp.size(), left.mat().size());
}

TEST(StereoBMTest, HasValidPixels) {
    auto [left, right] = make_stereo_pair({9,6}, 20, 16);
    cv::Mat disp = StereoBM{}(left, right);
    double min_val, max_val;
    cv::minMaxLoc(disp, &min_val, &max_val);
    EXPECT_GT(max_val, static_cast<double>(cv::StereoBM::DISP_SCALE * -1));
}

TEST(StereoBMTest, FluentSettersReturnThis) {
    StereoBM op;
    EXPECT_EQ(&op.num_disparities(32), &op);
    EXPECT_EQ(&op.block_size(11), &op);
}

// ── StereoSGBM ────────────────────────────────────────────────────────────────

TEST(StereoSGBMTest, ThrowsOnSizeMismatch) {
    auto [left, right] = make_stereo_pair({9,6}, 20, 16);
    cv::Mat smaller(left.mat().rows / 2, left.mat().cols / 2, CV_8UC1);
    EXPECT_THROW(StereoSGBM{}(left, Image<Gray>(smaller)), std::invalid_argument);
}

TEST(StereoSGBMTest, OutputIsCV16SAndMatchesInputSize) {
    auto [left, right] = make_stereo_pair({9,6}, 20, 16);
    cv::Mat disp = StereoSGBM{}(left, right);
    EXPECT_EQ(disp.type(), CV_16S);
    EXPECT_EQ(disp.size(), left.mat().size());
}

TEST(StereoSGBMTest, HasValidPixels) {
    auto [left, right] = make_stereo_pair({9,6}, 20, 16);
    cv::Mat disp = StereoSGBM{}(left, right);
    double min_val, max_val;
    cv::minMaxLoc(disp, &min_val, &max_val);
    EXPECT_GT(max_val, static_cast<double>(cv::StereoBM::DISP_SCALE * -1));
}

TEST(StereoSGBMTest, FluentSettersReturnThis) {
    StereoSGBM op;
    EXPECT_EQ(&op.min_disparity(0), &op);
    EXPECT_EQ(&op.num_disparities(64), &op);
    EXPECT_EQ(&op.block_size(3), &op);
    EXPECT_EQ(&op.p1(0), &op);
    EXPECT_EQ(&op.p2(0), &op);
    EXPECT_EQ(&op.mode(cv::StereoSGBM::MODE_SGBM), &op);
}
