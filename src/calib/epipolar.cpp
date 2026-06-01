// src/calib/epipolar.cpp
#include "improc/calib/ops/epipolar.hpp"
#include "improc/exceptions.hpp"
#include <opencv2/calib3d.hpp>

namespace improc::calib {

FundamentalMatResult FindFundamentalMat::operator()(
        const std::vector<cv::Point2f>& pts1,
        const std::vector<cv::Point2f>& pts2) const {
    if (pts1.size() != pts2.size())
        throw improc::ParameterError{"pts1", "must be same size as pts2", "FindFundamentalMat"};
    if (pts1.size() < 8)
        throw improc::ParameterError{"pts1", "at least 8 point correspondences required", "FindFundamentalMat"};

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
        throw improc::ParameterError{"pts1", "must be same size as pts2", "FindEssentialMat"};
    if (pts1.size() < 5)
        throw improc::ParameterError{"pts1", "at least 5 point correspondences required", "FindEssentialMat"};

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

cv::Mat TriangulatePoints::operator()(
        const cv::Mat& P1, const cv::Mat& P2,
        const std::vector<cv::Point2f>& pts1,
        const std::vector<cv::Point2f>& pts2) const {
    cv::Mat pts4d;
    cv::triangulatePoints(P1, P2, pts1, pts2, pts4d);
    return pts4d;
}

} // namespace improc::calib
