// include/improc/core/ops/watershed.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Applies the Watershed algorithm to segment an image using a marker image.
 *
 * Markers must be a CV_32S matrix where non-zero values indicate known regions
 * and zeros indicate unknown regions to be filled. After the call, markers are
 * updated in-place: boundaries are marked with -1.
 *
 * @code
 * cv::Mat markers = cv::Mat::zeros(img.mat().size(), CV_32S);
 * markers.at<int>(50, 50) = 1;
 * markers.at<int>(200, 200) = 2;
 * Watershed()(img, markers);
 * @endcode
 */
struct Watershed {
    /// @brief Runs Watershed. Modifies markers in-place; boundary pixels are set to -1.
    void operator()(const Image<BGR>& img, cv::Mat& markers) const;
};

} // namespace improc::core
