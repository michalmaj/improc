// include/improc/calib/ops/calib_types.hpp
#pragma once
#include <vector>
#include <opencv2/core.hpp>

namespace improc::calib {

struct FindChessboardResult {
    bool found = false;
    std::vector<cv::Point2f> corners;
};

struct CalibrationResult {
    cv::Mat camera_matrix;
    cv::Mat dist_coeffs;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    double rms = 0.0;
};

struct UndistortMapResult {
    cv::Mat map1; ///< x-map (CV_32FC1); use with Remap
    cv::Mat map2; ///< y-map (CV_32FC1); use with Remap
};

struct PnPResult {
    bool success = false;
    cv::Mat rvec;
    cv::Mat tvec;
};

struct PnPRansacResult {
    bool success = false;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat inliers; ///< CV_32S — indices of inlier correspondences
};

} // namespace improc::calib
