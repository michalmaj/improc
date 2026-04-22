#pragma once
#include "improc/core/image.hpp"
#include "improc/core/format_traits.hpp"
#include "improc/core/convert.hpp"

namespace improc::core {

/**
 * @brief Pipeline op: converts BGR image to HSV color space.
 *
 * @code
 * Image<HSV> hsv = bgr_img | ToHSV{};
 * @endcode
 */
struct ToHSV {
    Image<HSV> operator()(Image<BGR> img) const;
};

} // namespace improc::core
