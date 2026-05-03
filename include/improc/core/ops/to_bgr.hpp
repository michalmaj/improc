#pragma once
#include "improc/core/image.hpp"
#include "improc/core/format_traits.hpp"
#include "improc/core/convert.hpp"

namespace improc::core {

/**
 * @brief Pipeline op: converts Gray or HSV image to BGR color space.
 *
 * @code
 * Image<BGR> bgr = hsv_img | ToBGR{};
 * Image<BGR> bgr2 = gray_img | ToBGR{};
 * @endcode
 */
struct ToBGR {
    Image<BGR> operator()(Image<Gray> img) const;
    Image<BGR> operator()(Image<HSV>  img) const;
    Image<BGR> operator()(Image<LAB>   img) const { return convert<BGR, LAB>  (std::move(img)); }
    Image<BGR> operator()(Image<YCrCb> img) const { return convert<BGR, YCrCb>(std::move(img)); }
};

} // namespace improc::core
