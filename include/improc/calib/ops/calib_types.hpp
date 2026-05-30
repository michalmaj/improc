// include/improc/calib/ops/calib_types.hpp
#pragma once
#include <vector>
#include <opencv2/core.hpp>

namespace improc::calib {

/**
 * @brief Result of `cv::findChessboardCorners`.
 */
struct FindChessboardResult {
    bool found = false;
    std::vector<cv::Point2f> corners;
};

/**
 * @brief Output of `CalibrateCamera`: intrinsics, distortion, per-view poses, and RMS error.
 */
struct CalibrationResult {
    cv::Mat camera_matrix;         ///< 3×3 CV_64F camera intrinsic matrix.
    cv::Mat dist_coeffs;           ///< Distortion coefficients (OpenCV convention).
    std::vector<cv::Mat> rvecs;    ///< Per-view rotation vectors.
    std::vector<cv::Mat> tvecs;    ///< Per-view translation vectors.
    double rms = 0.0;              ///< RMS reprojection error in pixels.
};

/**
 * @brief Undistortion maps produced by `cv::initUndistortRectifyMap`.
 * Pass both maps to `cv::remap` (or `improc::core::Remap`) to undistort images.
 */
struct UndistortMapResult {
    cv::Mat map1; ///< x-map (CV_32FC1); pass to `Remap`.
    cv::Mat map2; ///< y-map (CV_32FC1); pass to `Remap`.
};

/**
 * @brief Result of `SolvePnP`: success flag and pose vectors.
 */
struct PnPResult {
    bool success = false; ///< True if PnP converged.
    cv::Mat rvec;         ///< 3×1 rotation vector (Rodrigues).
    cv::Mat tvec;         ///< 3×1 translation vector.
};

/**
 * @brief Result of `SolvePnPRansac`: pose vectors and inlier mask.
 */
struct PnPRansacResult {
    bool success = false; ///< True if RANSAC found a valid pose.
    cv::Mat rvec;         ///< 3×1 rotation vector (Rodrigues).
    cv::Mat tvec;         ///< 3×1 translation vector.
    cv::Mat inliers;      ///< CV_32S — indices of inlier correspondences.
};

/**
 * @brief Output of `StereoCalibrate`: per-camera intrinsics, relative pose, and epipolar matrices.
 */
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

/**
 * @brief Output of `StereoRectify`: rectification transforms, projection matrices, and Q matrix.
 */
struct StereoRectifyResult {
    cv::Mat R1, R2;        ///< 3×3 rectification transforms
    cv::Mat P1, P2;        ///< 3×4 projection matrices in rectified space
    cv::Mat Q;             ///< 4×4 disparity-to-depth mapping matrix
    cv::Rect validROI1;    ///< valid pixel region after rectification, left
    cv::Rect validROI2;    ///< valid pixel region after rectification, right
};

/**
 * @brief Fundamental matrix F and inlier mask from `FindFundamentalMat`.
 */
struct FundamentalMatResult {
    cv::Mat F;    ///< 3×3 CV_64F fundamental matrix
    cv::Mat mask; ///< CV_8U — 1 = inlier, 0 = outlier
};

/**
 * @brief Essential matrix E and inlier mask from `FindEssentialMat`.
 */
struct EssentialMatResult {
    cv::Mat E;    ///< 3×3 CV_64F essential matrix
    cv::Mat mask; ///< CV_8U — 1 = inlier, 0 = outlier
};

/**
 * @brief Rotation and (unit) translation recovered from an essential matrix.
 */
struct RecoverPoseResult {
    cv::Mat R;        ///< 3×3 rotation matrix.
    cv::Mat t;        ///< 3×1 translation vector (unit length).
    int inliers = 0;  ///< Number of inlier correspondences.
};

/**
 * @brief Detected ArUco markers: corner positions, IDs, and rejected candidates.
 */
struct ArucoResult {
    std::vector<std::vector<cv::Point2f>> corners;  ///< 4 corners per detected marker
    std::vector<int>                      ids;      ///< detected marker IDs (one per corners entry)
    std::vector<std::vector<cv::Point2f>> rejected; ///< corner candidates that failed validation
};

/**
 * @brief Per-marker pose: ID, rotation vector, and translation vector.
 */
struct ArucoPoseResult {
    int     id;   ///< marker ID (from ArucoResult)
    cv::Mat rvec; ///< 3×1 rotation vector
    cv::Mat tvec; ///< 3×1 translation vector
};

/**
 * @brief Result of `CharucoBoard`: ChArUco corners and detected ArUco markers.
 */
struct CharucoResult {
    std::vector<cv::Point2f>              charuco_corners; ///< interpolated ChArUco board corners
    std::vector<int>                      charuco_ids;     ///< board corner IDs (per charuco_corners entry)
    std::vector<std::vector<cv::Point2f>> marker_corners;  ///< detected ArUco marker corners
    std::vector<int>                      marker_ids;      ///< detected marker IDs
};

} // namespace improc::calib
