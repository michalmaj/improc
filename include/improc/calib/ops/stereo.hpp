// include/improc/calib/ops/stereo.hpp
#pragma once
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include "improc/calib/ops/calib_types.hpp"

namespace improc::calib {

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

} // namespace improc::calib
