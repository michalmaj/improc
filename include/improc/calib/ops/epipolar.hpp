// include/improc/calib/ops/epipolar.hpp
#pragma once
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>  // cv::FM_RANSAC, cv::RANSAC used in default member initializers
#include "improc/calib/ops/calib_types.hpp"

namespace improc::calib {

// ── FindFundamentalMat ────────────────────────────────────────────────────────

struct FindFundamentalMat {
    FindFundamentalMat& method(int m)              { method_           = m; return *this; }
    FindFundamentalMat& ransac_threshold(double t) { ransac_threshold_ = t; return *this; }
    FindFundamentalMat& confidence(double c)       { confidence_       = c; return *this; }

    // Throws std::invalid_argument if pts1.size() != pts2.size() or < 8 points.
    FundamentalMatResult operator()(const std::vector<cv::Point2f>& pts1,
                                    const std::vector<cv::Point2f>& pts2) const;

private:
    int    method_           = cv::FM_RANSAC;
    double ransac_threshold_ = 3.0;
    double confidence_       = 0.99;
};

// ── FindEssentialMat ──────────────────────────────────────────────────────────

struct FindEssentialMat {
    FindEssentialMat& method(int m)       { method_     = m; return *this; }
    FindEssentialMat& threshold(double t) { threshold_  = t; return *this; }
    FindEssentialMat& confidence(double c){ confidence_ = c; return *this; }

    // K: 3×3 camera matrix. Throws if sizes mismatch or < 5 points.
    EssentialMatResult operator()(const std::vector<cv::Point2f>& pts1,
                                  const std::vector<cv::Point2f>& pts2,
                                  const cv::Mat& K) const;

private:
    int    method_     = cv::RANSAC;
    double threshold_  = 1.0;
    double confidence_ = 0.99;
};

// ── RecoverPose ───────────────────────────────────────────────────────────────

struct RecoverPose {
    // E: essential matrix; pts1/pts2: same correspondences used to compute E.
    // K: camera matrix (same for both views if shared intrinsics).
    RecoverPoseResult operator()(const cv::Mat& E,
                                 const std::vector<cv::Point2f>& pts1,
                                 const std::vector<cv::Point2f>& pts2,
                                 const cv::Mat& K) const;
};

} // namespace improc::calib
