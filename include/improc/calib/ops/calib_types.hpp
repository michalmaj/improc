// include/improc/calib/ops/calib_types.hpp
#pragma once
#include <vector>
#include <opencv2/core.hpp>

namespace improc::calib {

struct FindChessboardResult {
    std::vector<cv::Point2f> corners;
    bool found;
};

struct CalibrationResult {
    cv::Mat K;
    cv::Mat dist;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    double rms;
};

struct UndistortMapResult {
    cv::Mat map1;
    cv::Mat map2;
};

struct PnPResult {
    cv::Mat rvec;
    cv::Mat tvec;
    bool success;
};

struct PnPRansacResult {
    cv::Mat rvec;
    cv::Mat tvec;
    bool success;
    cv::Mat inliers; // CV_32S — indices of inlier correspondences
};

} // namespace improc::calib
