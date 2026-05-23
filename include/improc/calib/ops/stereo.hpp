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

struct StereoCalibrate {
    // K1/dist1/K2/dist2 are optional initial estimates; if not set OpenCV initialises them.
    // Set flags = cv::CALIB_FIX_INTRINSIC when providing pre-calibrated intrinsics.
    StereoCalibrate& K1(cv::Mat k)    { K1_    = std::move(k); return *this; }
    StereoCalibrate& dist1(cv::Mat d) { dist1_ = std::move(d); return *this; }
    StereoCalibrate& K2(cv::Mat k)    { K2_    = std::move(k); return *this; }
    StereoCalibrate& dist2(cv::Mat d) { dist2_ = std::move(d); return *this; }
    StereoCalibrate& flags(int f)     { flags_ = f;             return *this; }

    // Throws std::invalid_argument if:
    //   - obj_pts / img_pts1 / img_pts2 sizes differ
    //   - fewer than 3 views
    //   - image_size dimensions not positive
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

struct StereoRectify {
    StereoRectify& alpha(double a)            { alpha_          = a; return *this; }
    StereoRectify& new_image_size(cv::Size s) { new_image_size_ = s; return *this; }

    // Throws std::invalid_argument if any of K1/dist1/K2/dist2/R/T is empty.
    StereoRectifyResult operator()(const cv::Mat& K1, const cv::Mat& dist1,
                                   const cv::Mat& K2, const cv::Mat& dist2,
                                   const cv::Mat& R,  const cv::Mat& T,
                                   cv::Size image_size) const;

private:
    double   alpha_          = -1.0;   // -1 = OpenCV default (all valid pixels)
    cv::Size new_image_size_ = {0, 0}; // {0,0} = same as input
};

// ── StereoBM ──────────────────────────────────────────────────────────────────

struct StereoBM {
    StereoBM& num_disparities(int n) { num_disparities_ = n; return *this; }
    StereoBM& block_size(int s)      { block_size_      = s; return *this; }

    // Returns CV_16S disparity map; divide by 16.0 for actual disparity values.
    // Throws std::invalid_argument if left/right sizes differ.
    cv::Mat operator()(Image<Gray> left, Image<Gray> right) const;

private:
    int num_disparities_ = 16;
    int block_size_      = 15;
};

// ── StereoSGBM ────────────────────────────────────────────────────────────────

struct StereoSGBM {
    StereoSGBM& min_disparity(int n)   { min_disparity_   = n; return *this; }
    StereoSGBM& num_disparities(int n) { num_disparities_ = n; return *this; }
    StereoSGBM& block_size(int s)      { block_size_      = s; return *this; }
    StereoSGBM& p1(int p)              { p1_              = p; return *this; }
    StereoSGBM& p2(int p)              { p2_              = p; return *this; }
    StereoSGBM& mode(int m)            { mode_            = m; return *this; }

    // Returns CV_16S disparity map; divide by 16.0 for actual disparity values.
    // Throws std::invalid_argument if left/right sizes differ.
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

struct ReprojectTo3D {
    ReprojectTo3D& handle_missing(bool h) { handle_missing_ = h; return *this; }

    // disparity: CV_16S or CV_32F; Q: 4×4 matrix from StereoRectify.
    // Returns CV_32FC3 point cloud; same size as disparity.
    cv::Mat operator()(const cv::Mat& disparity, const cv::Mat& Q) const;

private:
    bool handle_missing_ = false;
};

} // namespace improc::calib
