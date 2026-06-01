// src/core/ops/grabcut.cpp
#include "improc/core/ops/grabcut.hpp"
#include "improc/exceptions.hpp"
#include <stdexcept>

namespace improc::core {

GrabCut& GrabCut::iterations(int n) {
    iterations_ = n;
    return *this;
}

Image<Gray> GrabCut::operator()(const Image<BGR>& img, cv::Rect roi) const {
    if (roi.empty())
        throw improc::ParameterError{"roi", "must not be empty", "GrabCut"};
    if (roi.x < 0 || roi.y < 0 ||
        roi.x + roi.width  > img.cols() ||
        roi.y + roi.height > img.rows())
        throw improc::ParameterError{"roi", "extends outside image bounds", "GrabCut"};

    cv::Mat mask;
    cv::Mat bgModel, fgModel;
    cv::grabCut(img.mat(), mask, roi, bgModel, fgModel, iterations_,
                cv::GC_INIT_WITH_RECT);
    return Image<Gray>(mask);
}

} // namespace improc::core
