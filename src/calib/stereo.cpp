// src/calib/stereo.cpp
#include "improc/calib/ops/stereo.hpp"
#include <opencv2/calib3d.hpp>

namespace improc::calib {

StereoCalibrationResult StereoCalibrate::operator()(
        const std::vector<std::vector<cv::Point3f>>& obj_pts,
        const std::vector<std::vector<cv::Point2f>>& img_pts1,
        const std::vector<std::vector<cv::Point2f>>& img_pts2,
        cv::Size image_size) const {
    if (obj_pts.size() != img_pts1.size() || obj_pts.size() != img_pts2.size())
        throw std::invalid_argument(
            "StereoCalibrate: obj_pts, img_pts1, img_pts2 must have the same number of views");
    if (obj_pts.size() < 3)
        throw std::invalid_argument(
            "StereoCalibrate: at least 3 views required");
    if (image_size.width <= 0 || image_size.height <= 0)
        throw std::invalid_argument(
            "StereoCalibrate: image_size dimensions must be positive");

    StereoCalibrationResult result;
    result.K1    = K1_.empty()    ? cv::Mat() : K1_.clone();
    result.dist1 = dist1_.empty() ? cv::Mat() : dist1_.clone();
    result.K2    = K2_.empty()    ? cv::Mat() : K2_.clone();
    result.dist2 = dist2_.empty() ? cv::Mat() : dist2_.clone();

    result.rms = cv::stereoCalibrate(
        obj_pts, img_pts1, img_pts2,
        result.K1, result.dist1, result.K2, result.dist2,
        image_size, result.R, result.T, result.E, result.F,
        flags_);
    return result;
}

} // namespace improc::calib
