// include/improc/calib/ops/fisheye.hpp
#pragma once
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/calib/ops/calib_types.hpp"

namespace improc::calib {

using improc::core::Image;
using improc::core::AnyFormat;

// ── FisheyeCalibrate ──────────────────────────────────────────────────────────

/**
 * @brief Calibrates a fisheye camera from multi-view point correspondences.
 *
 * @code
 * CalibrationResult r = FisheyeCalibrate{}(obj_pts, img_pts, image_size);
 * @endcode
 */
struct FisheyeCalibrate {
    /// @brief Sets OpenCV fisheye calibration flags.
    FisheyeCalibrate& flags(int f) { flags_ = f; return *this; }
    /// @brief Sets termination criteria for the iterative optimisation.
    FisheyeCalibrate& criteria(cv::TermCriteria c) { criteria_ = c; return *this; }

    [[nodiscard]] CalibrationResult operator()(
        const std::vector<std::vector<cv::Point3f>>& object_pts,
        const std::vector<std::vector<cv::Point2f>>& image_pts,
        cv::Size image_size) const;

private:
    int flags_ = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
                 cv::fisheye::CALIB_CHECK_COND |
                 cv::fisheye::CALIB_FIX_SKEW;
    cv::TermCriteria criteria_{cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-6};
};

// ── FisheyeUndistort ──────────────────────────────────────────────────────────

/**
 * @brief Undistorts an image captured with a fisheye lens.
 *
 * @code
 * Image<BGR> out = FisheyeUndistort{}.K(K).dist(D)(img);
 * @endcode
 */
struct FisheyeUndistort {
    /// @brief Sets the camera intrinsic matrix.
    FisheyeUndistort& K(cv::Mat k)       { K_    = std::move(k); return *this; }
    /// @brief Sets the fisheye distortion coefficients (4×1 or 1×4).
    FisheyeUndistort& dist(cv::Mat d)    { dist_ = std::move(d); return *this; }
    /// @brief Sets the new (output) camera matrix; defaults to K if not set.
    FisheyeUndistort& new_K(cv::Mat nk)  { new_K_= std::move(nk); return *this; }

    template<AnyFormat F>
    [[nodiscard]] Image<F> operator()(Image<F> img) const {
        cv::Mat dst;
        cv::fisheye::undistortImage(img.mat(), dst, K_, dist_,
                                    new_K_.empty() ? K_ : new_K_);
        return Image<F>(std::move(dst));
    }

private:
    cv::Mat K_, dist_, new_K_;
};

// ── FisheyeUndistortPoints ────────────────────────────────────────────────────

/**
 * @brief Undistorts 2-D image points from a fisheye lens.
 *
 * @code
 * auto out = FisheyeUndistortPoints{}.K(K).dist(D)(pts);
 * @endcode
 */
struct FisheyeUndistortPoints {
    /// @brief Sets the camera intrinsic matrix.
    FisheyeUndistortPoints& K(cv::Mat k)    { K_    = std::move(k); return *this; }
    /// @brief Sets the fisheye distortion coefficients.
    FisheyeUndistortPoints& dist(cv::Mat d) { dist_ = std::move(d); return *this; }
    /// @brief Optional rectification matrix R.
    FisheyeUndistortPoints& R(cv::Mat r)    { R_    = std::move(r); return *this; }
    /// @brief Optional new projection matrix P.
    FisheyeUndistortPoints& P(cv::Mat p)    { P_    = std::move(p); return *this; }

    [[nodiscard]] std::vector<cv::Point2f> operator()(
        const std::vector<cv::Point2f>& pts) const;

private:
    cv::Mat K_, dist_, R_, P_;
};

// ── FisheyeInitRectifyMap ─────────────────────────────────────────────────────

/**
 * @brief Computes the undistortion and rectification maps for a fisheye lens.
 *
 * Pass the resulting maps to `cv::remap` (or `improc::core::Remap`).
 *
 * @code
 * UndistortMapResult maps = FisheyeInitRectifyMap{}.K(K).dist(D)(size);
 * @endcode
 */
struct FisheyeInitRectifyMap {
    /// @brief Sets the camera intrinsic matrix.
    FisheyeInitRectifyMap& K(cv::Mat k)      { K_     = std::move(k); return *this; }
    /// @brief Sets the fisheye distortion coefficients.
    FisheyeInitRectifyMap& dist(cv::Mat d)   { dist_  = std::move(d); return *this; }
    /// @brief Optional rectification matrix (defaults to identity).
    FisheyeInitRectifyMap& R(cv::Mat r)      { R_     = std::move(r); return *this; }
    /// @brief Optional new projection matrix (defaults to K).
    FisheyeInitRectifyMap& new_K(cv::Mat nk) { new_K_ = std::move(nk); return *this; }

    [[nodiscard]] UndistortMapResult operator()(cv::Size image_size) const;

private:
    cv::Mat K_, dist_, R_, new_K_;
};

// ── FisheyeStereoCalibrate ────────────────────────────────────────────────────

/**
 * @brief Calibrates a stereo fisheye rig from multi-view point correspondences.
 *
 * @code
 * StereoCalibrationResult r = FisheyeStereoCalibrate{}
 *     .K1(K1).dist1(D1).K2(K2).dist2(D2)
 *     (obj_pts, left_pts, right_pts, image_size);
 * @endcode
 */
struct FisheyeStereoCalibrate {
    /// @brief Sets initial left camera matrix.
    FisheyeStereoCalibrate& K1(cv::Mat k)    { K1_    = std::move(k); return *this; }
    /// @brief Sets initial left distortion coefficients.
    FisheyeStereoCalibrate& dist1(cv::Mat d) { dist1_ = std::move(d); return *this; }
    /// @brief Sets initial right camera matrix.
    FisheyeStereoCalibrate& K2(cv::Mat k)    { K2_    = std::move(k); return *this; }
    /// @brief Sets initial right distortion coefficients.
    FisheyeStereoCalibrate& dist2(cv::Mat d) { dist2_ = std::move(d); return *this; }
    /// @brief Sets OpenCV fisheye stereo calibration flags.
    FisheyeStereoCalibrate& flags(int f)     { flags_ = f; return *this; }

    [[nodiscard]] StereoCalibrationResult operator()(
        const std::vector<std::vector<cv::Point3f>>& object_pts,
        const std::vector<std::vector<cv::Point2f>>& left_pts,
        const std::vector<std::vector<cv::Point2f>>& right_pts,
        cv::Size image_size) const;

private:
    cv::Mat K1_, dist1_, K2_, dist2_;
    int flags_ = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
                 cv::fisheye::CALIB_CHECK_COND |
                 cv::fisheye::CALIB_FIX_SKEW;
};

// ── FisheyeStereoRectify ──────────────────────────────────────────────────────

/**
 * @brief Computes stereo rectification for a fisheye stereo rig.
 *
 * @code
 * StereoRectifyResult r = FisheyeStereoRectify{}
 *     .image_size({640,480})
 *     (K1, D1, K2, D2, R, T);
 * @endcode
 */
struct FisheyeStereoRectify {
    /// @brief Sets the image size used for rectification.
    FisheyeStereoRectify& image_size(cv::Size s) { image_size_ = s; return *this; }
    /// @brief Sets rectification flags (default: cv::CALIB_ZERO_DISPARITY).
    FisheyeStereoRectify& flags(int f)           { flags_ = f; return *this; }
    /// @brief Sets the balance parameter in [0, 1] (default: 0.0).
    FisheyeStereoRectify& balance(double b)      { balance_ = b; return *this; }
    /// @brief Sets the FOV scale factor (default: 1.0).
    FisheyeStereoRectify& fov_scale(double f)    { fov_scale_ = f; return *this; }

    [[nodiscard]] StereoRectifyResult operator()(
        const cv::Mat& K1, const cv::Mat& D1,
        const cv::Mat& K2, const cv::Mat& D2,
        const cv::Mat& R,  const cv::Mat& T) const;

private:
    cv::Size image_size_{640, 480};
    int      flags_     = cv::CALIB_ZERO_DISPARITY;
    double   balance_   = 0.0;
    double   fov_scale_ = 1.0;
};

} // namespace improc::calib
