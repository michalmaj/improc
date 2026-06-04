// src/core/ops/feature_detection.cpp
#include "improc/core/ops/feature_detection.hpp"
#include <opencv2/imgproc.hpp>

namespace improc::core {

namespace {
    cv::Mat to_gray(const cv::Mat& bgr) {
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
}

KeypointSet DetectORB::operator()(Image<Gray> img) const {
    KeypointSet result;
    cv::ORB::create(max_features_, scale_factor_, n_levels_)
        ->detect(img.mat(), result.keypoints);
    return result;
}

KeypointSet DetectSIFT::operator()(Image<Gray> img) const {
    KeypointSet result;
    cv::SIFT::create(max_features_, n_octave_layers_)
        ->detect(img.mat(), result.keypoints);
    return result;
}

KeypointSet DetectAKAZE::operator()(Image<Gray> img) const {
    KeypointSet result;
    cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, threshold_)
        ->detect(img.mat(), result.keypoints);
    return result;
}

DescriptorSet DescribeORB::operator()(Image<Gray> img) const {
    DescriptorSet result;
    result.keypoints = kps_;
    cv::ORB::create()->compute(img.mat(), result.keypoints.keypoints, result.descriptors);
    return result;
}

DescriptorSet DescribeORB::operator()(Image<BGR> img) const {
    DescriptorSet result;
    result.keypoints = kps_;
    cv::ORB::create()->compute(to_gray(img.mat()), result.keypoints.keypoints, result.descriptors);
    return result;
}

DescriptorSet DescribeSIFT::operator()(Image<Gray> img) const {
    DescriptorSet result;
    result.keypoints = kps_;
    cv::SIFT::create()->compute(img.mat(), result.keypoints.keypoints, result.descriptors);
    return result;
}

DescriptorSet DescribeSIFT::operator()(Image<BGR> img) const {
    DescriptorSet result;
    result.keypoints = kps_;
    cv::SIFT::create()->compute(to_gray(img.mat()), result.keypoints.keypoints, result.descriptors);
    return result;
}

DescriptorSet DescribeAKAZE::operator()(Image<Gray> img) const {
    DescriptorSet result;
    result.keypoints = kps_;
    cv::AKAZE::create()->compute(
        img.mat(), result.keypoints.keypoints, result.descriptors);
    return result;
}

DescriptorSet DescribeAKAZE::operator()(Image<BGR> img) const {
    DescriptorSet result;
    result.keypoints = kps_;
    cv::AKAZE::create()->compute(
        to_gray(img.mat()), result.keypoints.keypoints, result.descriptors);
    return result;
}

// ── BRISK ────────────────────────────────────────────────────────────────────

KeypointSet DetectBRISK::operator()(Image<Gray> img) const {
    KeypointSet result;
    cv::BRISK::create(threshold_, octaves_, pattern_scale_)
        ->detect(img.mat(), result.keypoints);
    return result;
}

DescriptorSet DescribeBRISK::operator()(Image<Gray> img) const {
    DescriptorSet result;
    result.keypoints = kps_;
    cv::BRISK::create(threshold_, octaves_, pattern_scale_)
        ->compute(img.mat(), result.keypoints.keypoints, result.descriptors);
    return result;
}

DescriptorSet DescribeBRISK::operator()(Image<BGR> img) const {
    DescriptorSet result;
    result.keypoints = kps_;
    cv::BRISK::create(threshold_, octaves_, pattern_scale_)
        ->compute(to_gray(img.mat()), result.keypoints.keypoints, result.descriptors);
    return result;
}

// ── KAZE ─────────────────────────────────────────────────────────────────────

KeypointSet DetectKAZE::operator()(Image<Gray> img) const {
    KeypointSet result;
    cv::KAZE::create(false, false, threshold_, octaves_, sublevels_)
        ->detect(img.mat(), result.keypoints);
    return result;
}

DescriptorSet DescribeKAZE::operator()(Image<Gray> img) const {
    DescriptorSet result;
    result.keypoints = kps_;
    cv::KAZE::create(false, false, threshold_, octaves_, sublevels_)
        ->compute(img.mat(), result.keypoints.keypoints, result.descriptors);
    return result;
}

DescriptorSet DescribeKAZE::operator()(Image<BGR> img) const {
    DescriptorSet result;
    result.keypoints = kps_;
    cv::KAZE::create(false, false, threshold_, octaves_, sublevels_)
        ->compute(to_gray(img.mat()), result.keypoints.keypoints, result.descriptors);
    return result;
}

std::vector<cv::Point2f> GoodFeaturesToTrack::operator()(const Image<Gray>& img) const {
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img.mat(), corners, max_corners_,
                            quality_level_, min_distance_,
                            cv::noArray(), 3, use_harris_);
    return corners;
}

} // namespace improc::core
