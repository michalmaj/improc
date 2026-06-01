// src/calib/project.cpp
#include "improc/calib/ops/project.hpp"
#include "improc/exceptions.hpp"
#include <opencv2/calib3d.hpp>

namespace improc::calib {

std::vector<cv::Point2f> ProjectPoints::operator()(
        const std::vector<cv::Point3f>& obj_pts,
        const cv::Mat& rvec, const cv::Mat& tvec,
        const cv::Mat& K, const cv::Mat& dist) const {
    if (obj_pts.empty())
        throw improc::ParameterError{"obj_pts", "must not be empty", "ProjectPoints"};
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(obj_pts, rvec, tvec, K, dist, img_pts);
    return img_pts;
}

PnPResult SolvePnP::operator()(const std::vector<cv::Point3f>& obj_pts,
                                const std::vector<cv::Point2f>& img_pts,
                                const cv::Mat& K, const cv::Mat& dist) const {
    if (obj_pts.size() != img_pts.size())
        throw improc::ParameterError{"obj_pts", "must be same size as img_pts", "SolvePnP"};
    if (obj_pts.size() < 4)
        throw improc::ParameterError{"obj_pts", "at least 4 correspondences required", "SolvePnP"};
    PnPResult result;
    result.success = cv::solvePnP(obj_pts, img_pts, K, dist,
                                  result.rvec, result.tvec,
                                  false, method_);
    return result;
}

PnPRansacResult SolvePnPRansac::operator()(const std::vector<cv::Point3f>& obj_pts,
                                            const std::vector<cv::Point2f>& img_pts,
                                            const cv::Mat& K, const cv::Mat& dist) const {
    if (obj_pts.size() != img_pts.size())
        throw improc::ParameterError{"obj_pts", "must be same size as img_pts", "SolvePnPRansac"};
    if (obj_pts.size() < 4)
        throw improc::ParameterError{"obj_pts", "at least 4 correspondences required", "SolvePnPRansac"};
    PnPRansacResult result;
    result.success = cv::solvePnPRansac(obj_pts, img_pts, K, dist,
                                        result.rvec, result.tvec,
                                        false, iterations_,
                                        reprojection_error_, confidence_,
                                        result.inliers, method_);
    return result;
}

} // namespace improc::calib
