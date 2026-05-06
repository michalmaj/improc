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

} // namespace improc::core
