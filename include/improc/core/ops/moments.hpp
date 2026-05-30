// include/improc/core/ops/moments.hpp
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Computes image moments (spatial, central, and Hu) for a grayscale image.
 */
struct Moments {
    bool binary = false; ///< If true, treats the image as binary before moment computation.

    /// @return cv::Moments containing spatial, central, and normalised central moments.
    cv::Moments operator()(const Image<Gray>& img) const {
        return cv::moments(img.mat(), binary);
    }
};

} // namespace improc::core
