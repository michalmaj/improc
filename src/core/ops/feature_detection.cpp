// src/core/ops/feature_detection.cpp
#include "improc/core/ops/feature_detection.hpp"

namespace improc::core {

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

} // namespace improc::core
