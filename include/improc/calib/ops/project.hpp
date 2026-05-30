// include/improc/calib/ops/project.hpp
#pragma once
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>  // cv::SOLVEPNP_ITERATIVE used in default member initializers
#include "improc/calib/ops/calib_types.hpp"

namespace improc::calib {

/**
 * @brief Projects 3-D object points onto a 2-D image plane.
 */
struct ProjectPoints {
    /// @brief Projects `obj_pts` using the given pose and camera parameters.
    /// @return Projected 2-D points in pixel coordinates.
    std::vector<cv::Point2f> operator()(const std::vector<cv::Point3f>& obj_pts,
                                        const cv::Mat& rvec,
                                        const cv::Mat& tvec,
                                        const cv::Mat& K,
                                        const cv::Mat& dist) const;
};

/**
 * @brief Solves the Perspective-n-Point problem to recover camera pose from 3-D/2-D correspondences.
 */
struct SolvePnP {
    /// @brief Sets the PnP solver algorithm (default: `cv::SOLVEPNP_ITERATIVE`).
    SolvePnP& method(int m) { method_ = m; return *this; }

    /// @brief Estimates the pose.
    /// @throws std::invalid_argument if sizes mismatch or fewer than 4 points.
    PnPResult operator()(const std::vector<cv::Point3f>& obj_pts,
                         const std::vector<cv::Point2f>& img_pts,
                         const cv::Mat& K,
                         const cv::Mat& dist) const;

private:
    int method_ = cv::SOLVEPNP_ITERATIVE;
};

/**
 * @brief RANSAC-robust variant of `SolvePnP` that also returns an inlier mask.
 */
struct SolvePnPRansac {
    /// @brief Sets the PnP solver algorithm (default: `cv::SOLVEPNP_ITERATIVE`).
    SolvePnPRansac& method(int m)               { method_             = m; return *this; }
    /// @brief Sets the required solution confidence in [0, 1] (default: 0.99).
    SolvePnPRansac& confidence(double c)        { confidence_         = c; return *this; }
    /// @brief Sets the maximum reprojection error (pixels) to count a point as inlier (default: 8.0).
    SolvePnPRansac& reprojection_error(float e) { reprojection_error_ = e; return *this; }
    /// @brief Sets the maximum number of RANSAC iterations (default: 100).
    SolvePnPRansac& iterations(int n)           { iterations_         = n; return *this; }

    /// @brief Estimates the pose with RANSAC.
    PnPRansacResult operator()(const std::vector<cv::Point3f>& obj_pts,
                               const std::vector<cv::Point2f>& img_pts,
                               const cv::Mat& K,
                               const cv::Mat& dist) const;

private:
    int    method_             = cv::SOLVEPNP_ITERATIVE;
    double confidence_         = 0.99;
    float  reprojection_error_ = 8.0f;
    int    iterations_         = 100;
};

} // namespace improc::calib
