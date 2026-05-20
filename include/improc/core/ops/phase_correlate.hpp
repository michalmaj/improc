// include/improc/core/ops/phase_correlate.hpp
#pragma once
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct PhaseCorrelateResult {
    cv::Point2d shift;
    double      response;
};

struct PhaseCorrelate {
    PhaseCorrelateResult operator()(const Image<Float32>& prev,
                                    const Image<Float32>& next) const;
};

} // namespace improc::core
