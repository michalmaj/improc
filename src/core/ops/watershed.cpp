// src/core/ops/watershed.cpp
#include "improc/core/ops/watershed.hpp"
#include <stdexcept>

namespace improc::core {

void Watershed::operator()(const Image<BGR>& img, cv::Mat& markers) const {
    if (markers.type() != CV_32SC1)
        throw std::invalid_argument("Watershed: markers must be CV_32SC1");
    if (markers.size() != img.mat().size())
        throw std::invalid_argument("Watershed: markers size must match image size");
    cv::watershed(img.mat(), markers);
}

} // namespace improc::core
