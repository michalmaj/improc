// include/improc/core/ops/moments.hpp
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct Moments {
    bool binary = false;

    cv::Moments operator()(const Image<Gray>& img) const {
        return cv::moments(img.mat(), binary);
    }
};

} // namespace improc::core
