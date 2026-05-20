// include/improc/core/ops/grabcut.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct GrabCut {
    GrabCut& iterations(int n);

    Image<Gray> operator()(const Image<BGR>& img, cv::Rect roi) const;

private:
    int iterations_{5};
};

} // namespace improc::core
