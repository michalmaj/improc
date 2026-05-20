// src/core/ops/phase_correlate.cpp
#include "improc/core/ops/phase_correlate.hpp"

namespace improc::core {

PhaseCorrelateResult PhaseCorrelate::operator()(const Image<Float32>& prev,
                                                 const Image<Float32>& next) const {
    if (prev.rows() != next.rows() || prev.cols() != next.cols())
        throw std::invalid_argument("PhaseCorrelate: prev and next must have the same size");
    cv::Mat hann;
    cv::createHanningWindow(hann, prev.mat().size(), CV_32F);
    double response;
    cv::Point2d shift = cv::phaseCorrelate(prev.mat(), next.mat(), hann, &response);
    return {shift, response};
}

} // namespace improc::core
