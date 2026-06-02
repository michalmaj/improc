// src/core/ops/watershed.cpp
#include "improc/core/ops/watershed.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

void Watershed::operator()(const Image<BGR>& img, cv::Mat& markers) const {
    if (markers.type() != CV_32SC1)
        throw improc::ParameterError{"markers", "must be CV_32SC1", "Watershed"};
    if (markers.size() != img.mat().size())
        throw improc::ParameterError{"markers", "size must match image size", "Watershed"};
    cv::watershed(img.mat(), markers);
}

} // namespace improc::core
