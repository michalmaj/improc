// include/improc/core/ops/hist_eq.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Global histogram equalization — redistributes pixel intensities across the full range.
 *
 * On `Image<Gray>`: calls `cv::equalizeHist` directly.
 * On `Image<BGR>`:  converts to YCrCb, equalizes the Y (luminance) channel only,
 *                   then converts back — preserving hue and saturation.
 *
 * No parameters. Stateless.
 *
 * @code
 * Image<Gray> eq_gray = gray | HistogramEqualization{};
 * Image<BGR>  eq_bgr  = bgr  | HistogramEqualization{};
 * @endcode
 */
struct HistogramEqualization {
    /// @brief Equalizes the histogram of a single-channel gray image.
    Image<Gray> operator()(Image<Gray> img) const;
    /// @brief Equalizes the luminance (Y) channel of a BGR image.
    Image<BGR>  operator()(Image<BGR>  img) const;
};

} // namespace improc::core
