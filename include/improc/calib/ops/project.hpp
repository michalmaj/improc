// include/improc/calib/ops/project.hpp
#pragma once
#include <stdexcept>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>  // cv::SOLVEPNP_ITERATIVE used in default member initializers
#include "improc/calib/ops/calib_types.hpp"

namespace improc::calib {

struct ProjectPoints {
    std::vector<cv::Point2f> operator()(const std::vector<cv::Point3f>& obj_pts,
                                        const cv::Mat& rvec,
                                        const cv::Mat& tvec,
                                        const cv::Mat& K,
                                        const cv::Mat& dist) const;
};

struct SolvePnP {
    SolvePnP& method(int m) { method_ = m; return *this; }

    // Throws std::invalid_argument if sizes mismatch or fewer than 4 points.
    PnPResult operator()(const std::vector<cv::Point3f>& obj_pts,
                         const std::vector<cv::Point2f>& img_pts,
                         const cv::Mat& K,
                         const cv::Mat& dist) const;

private:
    int method_ = cv::SOLVEPNP_ITERATIVE;
};

struct SolvePnPRansac {
    SolvePnPRansac& method(int m)               { method_             = m; return *this; }
    SolvePnPRansac& confidence(double c)        { confidence_         = c; return *this; }
    SolvePnPRansac& reprojection_error(float e) { reprojection_error_ = e; return *this; }
    SolvePnPRansac& iterations(int n)           { iterations_         = n; return *this; }

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
