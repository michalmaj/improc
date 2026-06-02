// include/improc/calib/ops/epipolar.hpp
#pragma once
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>  // cv::FM_RANSAC, cv::RANSAC used in default member initializers
#include "improc/calib/ops/calib_types.hpp"

namespace improc::calib {

// ── FindFundamentalMat ────────────────────────────────────────────────────────

/**
 * @brief Estimates the fundamental matrix from point correspondences using RANSAC.
 */
struct FindFundamentalMat {
    /// @brief Sets the estimation method (default: `cv::FM_RANSAC`).
    FindFundamentalMat& method(int m)              { method_           = m; return *this; }
    /// @brief Sets the RANSAC reprojection threshold in pixels (default: 3.0).
    FindFundamentalMat& ransac_threshold(double t) { ransac_threshold_ = t; return *this; }
    /// @brief Sets the desired solution confidence in [0, 1] (default: 0.99).
    FindFundamentalMat& confidence(double c)       { confidence_       = c; return *this; }

    /// @brief Estimates the fundamental matrix.
    /// @throws improc::ParameterError if `pts1.size() != pts2.size()` or fewer than 8 points.
    [[nodiscard]] FundamentalMatResult operator()(const std::vector<cv::Point2f>& pts1,
                                    const std::vector<cv::Point2f>& pts2) const;

private:
    int    method_           = cv::FM_RANSAC;
    double ransac_threshold_ = 3.0;
    double confidence_       = 0.99;
};

// ── FindEssentialMat ──────────────────────────────────────────────────────────

/**
 * @brief Estimates the essential matrix from point correspondences and a shared camera matrix.
 */
struct FindEssentialMat {
    /// @brief Sets the estimation method (default: `cv::RANSAC`).
    FindEssentialMat& method(int m)       { method_     = m; return *this; }
    /// @brief Sets the RANSAC reprojection threshold in pixels (default: 1.0).
    FindEssentialMat& threshold(double t) { threshold_  = t; return *this; }
    /// @brief Sets the desired solution confidence in [0, 1] (default: 0.99).
    FindEssentialMat& confidence(double c){ confidence_ = c; return *this; }

    /// @brief Estimates the essential matrix.
    /// @param K 3×3 camera intrinsic matrix (same for both views).
    /// @throws improc::ParameterError if sizes mismatch or fewer than 5 points.
    [[nodiscard]] EssentialMatResult operator()(const std::vector<cv::Point2f>& pts1,
                                  const std::vector<cv::Point2f>& pts2,
                                  const cv::Mat& K) const;

private:
    int    method_     = cv::RANSAC;
    double threshold_  = 1.0;
    double confidence_ = 0.99;
};

// ── RecoverPose ───────────────────────────────────────────────────────────────

/**
 * @brief Recovers rotation and (unit) translation from an essential matrix via cheirality check.
 */
struct RecoverPose {
    /// @brief Recovers R and t.
    /// @param E Essential matrix (from `FindEssentialMat`).
    /// @param K Camera matrix (shared intrinsics assumed).
    [[nodiscard]] RecoverPoseResult operator()(const cv::Mat& E,
                                 const std::vector<cv::Point2f>& pts1,
                                 const std::vector<cv::Point2f>& pts2,
                                 const cv::Mat& K) const;
};

// ── TriangulatePoints ─────────────────────────────────────────────────────────

/**
 * @brief Triangulates 3-D points from matched 2-D correspondences and two projection matrices.
 */
struct TriangulatePoints {
    /// @brief Triangulates `pts1`/`pts2` using `P1`/`P2`.
    /// @param P1 3×4 projection matrix of the first camera.
    /// @param P2 3×4 projection matrix of the second camera.
    /// @return 4×N homogeneous point cloud (CV_32F); divide by row 3 for Euclidean coordinates.
    [[nodiscard]] cv::Mat operator()(const cv::Mat& P1, const cv::Mat& P2,
                       const std::vector<cv::Point2f>& pts1,
                       const std::vector<cv::Point2f>& pts2) const;
};

} // namespace improc::calib
