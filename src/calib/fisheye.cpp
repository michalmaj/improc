// src/calib/fisheye.cpp
#include "improc/calib/ops/fisheye.hpp"
#include <opencv2/calib3d.hpp>

namespace improc::calib {

// ── FisheyeCalibrate ──────────────────────────────────────────────────────────

CalibrationResult FisheyeCalibrate::operator()(
    const std::vector<std::vector<cv::Point3f>>& object_pts,
    const std::vector<std::vector<cv::Point2f>>& image_pts,
    cv::Size image_size) const
{
    CalibrationResult r;
    r.rms = cv::fisheye::calibrate(
        object_pts, image_pts, image_size,
        r.camera_matrix, r.dist_coeffs, r.rvecs, r.tvecs,
        flags_, criteria_);
    return r;
}

// ── FisheyeUndistortPoints ────────────────────────────────────────────────────

std::vector<cv::Point2f> FisheyeUndistortPoints::operator()(
    const std::vector<cv::Point2f>& pts) const
{
    std::vector<cv::Point2f> result;
    cv::fisheye::undistortPoints(pts, result, K_, dist_,
                                 R_.empty() ? cv::Mat{} : R_,
                                 P_.empty() ? cv::Mat{} : P_);
    return result;
}

// ── FisheyeInitRectifyMap ─────────────────────────────────────────────────────

UndistortMapResult FisheyeInitRectifyMap::operator()(cv::Size image_size) const {
    UndistortMapResult r;
    cv::fisheye::initUndistortRectifyMap(
        K_, dist_,
        R_.empty() ? cv::Mat::eye(3, 3, CV_64F) : R_,
        new_K_.empty() ? K_ : new_K_,
        image_size, CV_32FC1, r.map1, r.map2);
    return r;
}

// ── FisheyeStereoCalibrate ────────────────────────────────────────────────────

StereoCalibrationResult FisheyeStereoCalibrate::operator()(
    const std::vector<std::vector<cv::Point3f>>& object_pts,
    const std::vector<std::vector<cv::Point2f>>& left_pts,
    const std::vector<std::vector<cv::Point2f>>& right_pts,
    cv::Size image_size) const
{
    StereoCalibrationResult result;
    result.K1    = K1_.empty()    ? cv::Mat() : K1_.clone();
    result.dist1 = dist1_.empty() ? cv::Mat() : dist1_.clone();
    result.K2    = K2_.empty()    ? cv::Mat() : K2_.clone();
    result.dist2 = dist2_.empty() ? cv::Mat() : dist2_.clone();

    // cv::fisheye::stereoCalibrate does not output E and F.
    // Use the overload without per-view rvecs/tvecs.
    result.rms = cv::fisheye::stereoCalibrate(
        object_pts, left_pts, right_pts,
        result.K1, result.dist1, result.K2, result.dist2,
        image_size, result.R, result.T,
        flags_,
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6));
    // E and F are not computed by the fisheye model; leave them empty.
    return result;
}

// ── FisheyeStereoRectify ──────────────────────────────────────────────────────

StereoRectifyResult FisheyeStereoRectify::operator()(
    const cv::Mat& K1, const cv::Mat& D1,
    const cv::Mat& K2, const cv::Mat& D2,
    const cv::Mat& R,  const cv::Mat& T) const
{
    StereoRectifyResult result;
    // cv::fisheye::stereoRectify(K1, D1, K2, D2, imageSize, R, tvec,
    //     R1, R2, P1, P2, Q, flags, newImageSize={0,0}, balance=0, fov_scale=1)
    cv::fisheye::stereoRectify(
        K1, D1, K2, D2, image_size_, R, T,
        result.R1, result.R2, result.P1, result.P2, result.Q,
        flags_, cv::Size{0, 0}, balance_, fov_scale_);
    return result;
}

} // namespace improc::calib
