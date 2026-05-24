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

struct StereoCalibrationResult {
    cv::Mat K1;    ///< 3×3 CV_64F — left camera intrinsics
    cv::Mat dist1; ///< distortion coefficients, left camera
    cv::Mat K2;    ///< 3×3 CV_64F — right camera intrinsics
    cv::Mat dist2; ///< distortion coefficients, right camera
    cv::Mat R;     ///< 3×3 rotation between cameras
    cv::Mat T;     ///< 3×1 translation between cameras
    cv::Mat E;     ///< 3×3 essential matrix
    cv::Mat F;     ///< 3×3 fundamental matrix
    double rms = 0.0;
};

struct StereoRectifyResult {
    cv::Mat R1, R2;        ///< 3×3 rectification transforms
    cv::Mat P1, P2;        ///< 3×4 projection matrices in rectified space
    cv::Mat Q;             ///< 4×4 disparity-to-depth mapping matrix
    cv::Rect validROI1;    ///< valid pixel region after rectification, left
    cv::Rect validROI2;    ///< valid pixel region after rectification, right
};

struct FundamentalMatResult {
    cv::Mat F;    ///< 3×3 CV_64F fundamental matrix
    cv::Mat mask; ///< CV_8U — 1 = inlier, 0 = outlier
};

struct EssentialMatResult {
    cv::Mat E;    ///< 3×3 CV_64F essential matrix
    cv::Mat mask; ///< CV_8U — 1 = inlier, 0 = outlier
};

struct RecoverPoseResult {
    cv::Mat R;        ///< 3×3 rotation matrix
    cv::Mat t;        ///< 3×1 translation vector (unit length)
    int inliers = 0;
};

struct ArucoResult {
    std::vector<std::vector<cv::Point2f>> corners; ///< 4 corners per detected marker
    std::vector<int>                      ids;
    std::vector<std::vector<cv::Point2f>> rejected;
};

struct ArucoPoseResult {
    int     id;   ///< marker ID (copied from ArucoResult)
    cv::Mat rvec; ///< 3×1 rotation vector
    cv::Mat tvec; ///< 3×1 translation vector
};

struct CharucoResult {
    std::vector<cv::Point2f>              charuco_corners;
    std::vector<int>                      charuco_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
    std::vector<int>                      marker_ids;
};

} // namespace improc::calib
