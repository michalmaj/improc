// src/ml/hog_detector.cpp
#include "improc/ml/hog_detector.hpp"
#include <cmath>

namespace improc::ml {

HOGDetector::HOGDetector() {
    hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
}

HOGDetector::HOGDetector(const std::vector<float>& svm) {
    if (svm.empty())
        throw ParameterError{"svm", "must not be empty", "HOGDetector"};
    hog_.setSVMDetector(svm);
}

std::vector<Detection> HOGDetector::operator()(const Image<BGR>& img) const {
    std::vector<cv::Rect>  locations;
    std::vector<double>    weights;

    hog_.detectMultiScale(img.mat(), locations, weights,
                          hit_threshold_, win_stride_, padding_, scale_);

    std::vector<Detection> result;
    result.reserve(locations.size());
    for (std::size_t i = 0; i < locations.size(); ++i) {
        // HOG weights are unbounded; sigmoid maps them to (0, 1)
        float conf = 1.0f / (1.0f + static_cast<float>(std::exp(-weights[i])));
        result.push_back({cv::Rect2f{locations[i]}, 0, conf, "person"});
    }
    return result;
}

} // namespace improc::ml
