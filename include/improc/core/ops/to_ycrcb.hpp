// include/improc/core/ops/to_ycrcb.hpp
#pragma once
#include "improc/core/image.hpp"
#include "improc/core/format_traits.hpp"
#include "improc/core/convert.hpp"

namespace improc::core {

/**
 * @brief Pipeline op: converts BGR image to YCrCb color space.
 *
 * Uses `cv::COLOR_BGR2YCrCb`. Output is `Image<YCrCb>` (CV_8UC3).
 * The Y channel carries luminance; Cr and Cb carry colour difference.
 * To convert back: `ycrcb_img | ToBGR{}`.
 *
 * @code
 * Image<YCrCb> y = bgr_img | ToYCrCb{};
 * @endcode
 */
struct ToYCrCb {
    Image<YCrCb> operator()(Image<BGR> img) const;
};

} // namespace improc::core
