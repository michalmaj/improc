// src/calib/epipolar.cpp
#include "improc/calib/ops/epipolar.hpp"
#include <opencv2/calib3d.hpp>

namespace improc::calib {

FundamentalMatResult FindFundamentalMat::operator()(
        const std::vector<cv::Point2f>& pts1,
        const std::vector<cv::Point2f>& pts2) const {
    if (pts1.size() != pts2.size())
        throw std::invalid_argument(
            "FindFundamentalMat: pts1 and pts2 must have the same number of points");
    if (pts1.size() < 8)
        throw std::invalid_argument(
            "FindFundamentalMat: at least 8 point correspondences required");

    FundamentalMatResult result;
    result.F = cv::findFundamentalMat(pts1, pts2, method_,
                                      ransac_threshold_, confidence_, result.mask);
    return result;
}

EssentialMatResult FindEssentialMat::operator()(
        const std::vector<cv::Point2f>& pts1,
        const std::vector<cv::Point2f>& pts2,
        const cv::Mat& K) const {
    if (pts1.size() != pts2.size())
        throw std::invalid_argument(
            "FindEssentialMat: pts1 and pts2 must have the same number of points");
    if (pts1.size() < 5)
        throw std::invalid_argument(
            "FindEssentialMat: at least 5 point correspondences required");

    EssentialMatResult result;
    result.E = cv::findEssentialMat(pts1, pts2, K, method_,
                                    confidence_, threshold_, result.mask);
    return result;
}

RecoverPoseResult RecoverPose::operator()(
        const cv::Mat& E,
        const std::vector<cv::Point2f>& pts1,
        const std::vector<cv::Point2f>& pts2,
        const cv::Mat& K) const {
    RecoverPoseResult result;
    result.inliers = cv::recoverPose(E, pts1, pts2, K, result.R, result.t);
    return result;
}

} // namespace improc::calib
