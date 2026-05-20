// src/core/ops/grabcut.cpp
#include "improc/core/ops/grabcut.hpp"
#include <stdexcept>

namespace improc::core {

GrabCut& GrabCut::iterations(int n) {
    iterations_ = n;
    return *this;
}

Image<Gray> GrabCut::operator()(const Image<BGR>& img, cv::Rect roi) const {
    if (roi.empty())
        throw std::invalid_argument("GrabCut: roi must not be empty");
    if (roi.x < 0 || roi.y < 0 ||
        roi.x + roi.width  > img.cols() ||
        roi.y + roi.height > img.rows())
        throw std::invalid_argument("GrabCut: roi extends outside image bounds");

    cv::Mat mask;
    cv::Mat bgModel, fgModel;
    cv::grabCut(img.mat(), mask, roi, bgModel, fgModel, iterations_,
                cv::GC_INIT_WITH_RECT);
    return Image<Gray>(mask);
}

} // namespace improc::core
