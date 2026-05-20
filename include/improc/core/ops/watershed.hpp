// include/improc/core/ops/watershed.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct Watershed {
    void operator()(const Image<BGR>& img, cv::Mat& markers) const;
};

} // namespace improc::core
