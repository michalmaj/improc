// include/improc/calib/ops/stereo.hpp
#pragma once
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>  // cv::StereoSGBM::MODE_SGBM used in default member initializer
#include "improc/calib/ops/calib_types.hpp"
#include "improc/core/image.hpp"

namespace improc::calib {

using improc::core::Image;
using improc::core::Gray;

// ── StereoCalibrate ───────────────────────────────────────────────────────────

/**
 * @brief Calibrates a stereo rig from multi-view chessboard correspondences.
 *
 * @code
 * auto result = StereoCalibrate()
 *     .flags(cv::CALIB_FIX_INTRINSIC)
 *     .K1(left_K).dist1(left_dist)
 *     .K2(right_K).dist2(right_dist)
 *     (obj_pts, img_pts_left, img_pts_right, image_size);
 * @endcode
 */
struct StereoCalibrate {
    /// @brief Sets initial left camera matrix (optional).
    StereoCalibrate& K1(cv::Mat k)    { K1_    = std::move(k); return *this; }
    /// @brief Sets initial left distortion coefficients (optional).
    StereoCalibrate& dist1(cv::Mat d) { dist1_ = std::move(d); return *this; }
    /// @brief Sets initial right camera matrix (optional).
    StereoCalibrate& K2(cv::Mat k)    { K2_    = std::move(k); return *this; }
    /// @brief Sets initial right distortion coefficients (optional).
    StereoCalibrate& dist2(cv::Mat d) { dist2_ = std::move(d); return *this; }
    /// @brief Sets OpenCV stereo calibration flags (default: 0).
    StereoCalibrate& flags(int f)     { flags_ = f;             return *this; }

    /// @brief Runs stereo calibration.
    /// @throws std::invalid_argument if point set sizes mismatch, fewer than 3 views, or image_size is non-positive.
    StereoCalibrationResult operator()(
        const std::vector<std::vector<cv::Point3f>>& obj_pts,
        const std::vector<std::vector<cv::Point2f>>& img_pts1,
        const std::vector<std::vector<cv::Point2f>>& img_pts2,
        cv::Size image_size) const;

private:
    cv::Mat K1_, dist1_, K2_, dist2_;
    int flags_ = 0;
};

// ── StereoRectify ─────────────────────────────────────────────────────────────

/**
 * @brief Computes rectification transforms, projection matrices, and disparity-to-depth matrix Q.
 */
struct StereoRectify {
    /// @brief Sets the free scaling parameter in [-1, 1] (default: -1 = OpenCV default).
    StereoRectify& alpha(double a)            { alpha_          = a; return *this; }
    /// @brief Sets the new image size after rectification (default: {0,0} = same as input).
    StereoRectify& new_image_size(cv::Size s) { new_image_size_ = s; return *this; }

    /// @brief Computes stereo rectification.
    /// @throws std::invalid_argument if any of the input matrices is empty.
    StereoRectifyResult operator()(const cv::Mat& K1, const cv::Mat& dist1,
                                   const cv::Mat& K2, const cv::Mat& dist2,
                                   const cv::Mat& R,  const cv::Mat& T,
                                   cv::Size image_size) const;

private:
    double   alpha_          = -1.0;   // -1 = OpenCV default (all valid pixels)
    cv::Size new_image_size_ = {0, 0}; // {0,0} = same as input
};

// ── StereoBM ──────────────────────────────────────────────────────────────────

/**
 * @brief Block-matching stereo disparity computation.
 * @return CV_16S disparity map. Divide values by 16 for actual disparity in pixels.
 */
struct StereoBM {
    /// @brief Sets the number of disparities (must be divisible by 16; default: 16).
    StereoBM& num_disparities(int n) { num_disparities_ = n; return *this; }
    /// @brief Sets the block size (must be odd; default: 15).
    StereoBM& block_size(int s)      { block_size_      = s; return *this; }

    /// @brief Computes the disparity map.
    /// @throws std::invalid_argument if `left` and `right` have different sizes.
    cv::Mat operator()(Image<Gray> left, Image<Gray> right) const;

private:
    int num_disparities_ = 16;
    int block_size_      = 15;
};

// ── StereoSGBM ────────────────────────────────────────────────────────────────

/**
 * @brief Semi-global block-matching stereo disparity computation (Hirschmuller).
 * @return CV_16S disparity map. Divide values by 16 for actual disparity in pixels.
 */
struct StereoSGBM {
    /// @brief Sets the minimum disparity value (default: 0).
    StereoSGBM& min_disparity(int n)   { min_disparity_   = n; return *this; }
    /// @brief Sets the disparity search range (must be divisible by 16; default: 64).
    StereoSGBM& num_disparities(int n) { num_disparities_ = n; return *this; }
    /// @brief Sets the matched block size (default: 3).
    StereoSGBM& block_size(int s)      { block_size_      = s; return *this; }
    /// @brief Sets P1 smoothness penalty (default: 0; recommended: 8×blockSize²).
    StereoSGBM& p1(int p)              { p1_              = p; return *this; }
    /// @brief Sets P2 smoothness penalty (default: 0; recommended: 32×blockSize²).
    StereoSGBM& p2(int p)              { p2_              = p; return *this; }
    /// @brief Sets the SGBM mode (default: `cv::StereoSGBM::MODE_SGBM`).
    StereoSGBM& mode(int m)            { mode_            = m; return *this; }

    /// @brief Computes the disparity map.
    /// @throws std::invalid_argument if `left` and `right` have different sizes.
    cv::Mat operator()(Image<Gray> left, Image<Gray> right) const;

private:
    int min_disparity_   = 0;
    int num_disparities_ = 64;
    int block_size_      = 3;
    int p1_              = 0;  // 0 = no smoothness; recommended: 8*blockSize²
    int p2_              = 0;  // 0 = no smoothness; recommended: 32*blockSize²
    int mode_            = cv::StereoSGBM::MODE_SGBM;
};

// ── ReprojectTo3D ─────────────────────────────────────────────────────────────

/**
 * @brief Reprojects a disparity map to a 3-D point cloud using the Q matrix from `StereoRectify`.
 */
struct ReprojectTo3D {
    /// @brief If true, handles missing (invalid) disparity values (default: false).
    ReprojectTo3D& handle_missing(bool h) { handle_missing_ = h; return *this; }

    /// @brief Reprojects disparity to 3-D.
    /// @param disparity CV_16S or CV_32F disparity map.
    /// @param Q         4×4 disparity-to-depth matrix from `StereoRectify`.
    /// @return CV_32FC3 point cloud; same size as `disparity`.
    cv::Mat operator()(const cv::Mat& disparity, const cv::Mat& Q) const;

private:
    bool handle_missing_ = false;
};

} // namespace improc::calib
