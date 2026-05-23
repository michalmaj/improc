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

StereoRectifyResult StereoRectify::operator()(
        const cv::Mat& K1, const cv::Mat& dist1,
        const cv::Mat& K2, const cv::Mat& dist2,
        const cv::Mat& R,  const cv::Mat& T,
        cv::Size image_size) const {
    if (K1.empty() || dist1.empty() || K2.empty() || dist2.empty() ||
        R.empty()  || T.empty())
        throw std::invalid_argument(
            "StereoRectify: K1, dist1, K2, dist2, R, T must not be empty");

    StereoRectifyResult result;
    cv::stereoRectify(K1, dist1, K2, dist2, image_size, R, T,
                      result.R1, result.R2, result.P1, result.P2, result.Q,
                      cv::CALIB_ZERO_DISPARITY, alpha_, new_image_size_,
                      &result.validROI1, &result.validROI2);
    return result;
}

cv::Mat StereoBM::operator()(Image<Gray> left, Image<Gray> right) const {
    if (left.mat().size() != right.mat().size())
        throw std::invalid_argument(
            "StereoBM: left and right images must have the same size");
    auto bm = cv::StereoBM::create(num_disparities_, block_size_);
    cv::Mat disparity;
    bm->compute(left.mat(), right.mat(), disparity);
    return disparity;
}

cv::Mat StereoSGBM::operator()(Image<Gray> left, Image<Gray> right) const {
    if (left.mat().size() != right.mat().size())
        throw std::invalid_argument(
            "StereoSGBM: left and right images must have the same size");
    auto sgbm = cv::StereoSGBM::create(min_disparity_, num_disparities_, block_size_,
                                        p1_, p2_, 0, 0, 0, 0, 0, mode_);
    cv::Mat disparity;
    sgbm->compute(left.mat(), right.mat(), disparity);
    return disparity;
}

} // namespace improc::calib
