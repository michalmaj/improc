// include/improc/core/ops/flood_fill.hpp
#pragma once
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Fills a connected region starting at a seed pixel with a new colour.
 *
 * lo_diff and up_diff define the pixel value tolerance for region membership.
 *
 * @code
 * auto filled = FloodFill().lo_diff({10,10,10}).up_diff({10,10,10})(img, {100, 100}, {0,0,255});
 * @endcode
 */
struct FloodFill {
    /// @brief Sets the lower boundary of the colour difference for region membership.
    FloodFill& lo_diff(cv::Scalar lo) { lo_diff_ = lo; return *this; }
    /// @brief Sets the upper boundary of the colour difference for region membership.
    FloodFill& up_diff(cv::Scalar hi) { up_diff_ = hi; return *this; }

    /// @brief Fills a connected region in a BGR image.
    Image<BGR>  operator()(const Image<BGR>&  img, cv::Point seed, cv::Scalar new_color) const;
    /// @brief Fills a connected region in a Gray image.
    Image<Gray> operator()(const Image<Gray>& img, cv::Point seed, uchar      new_val)   const;

private:
    cv::Scalar lo_diff_{0, 0, 0};
    cv::Scalar up_diff_{0, 0, 0};
};

} // namespace improc::core
