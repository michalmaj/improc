// include/improc/core/ops/to_lab.hpp
#pragma once
#include "improc/core/image.hpp"
#include "improc/core/format_traits.hpp"
#include "improc/core/convert.hpp"

namespace improc::core {

/**
 * @brief Pipeline op: converts BGR image to CIE L*a*b* color space.
 *
 * Uses `cv::COLOR_BGR2Lab`. Output is `Image<LAB>` (CV_8UC3).
 * To convert back: `lab_img | ToBGR{}`.
 *
 * @code
 * Image<LAB> lab = bgr_img | ToLAB{};
 * @endcode
 */
struct ToLAB {
    Image<LAB> operator()(Image<BGR> img) const;
};

} // namespace improc::core
