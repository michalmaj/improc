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
    Image<BGR> operator()(const Image<Gray>& img) const;
    Image<BGR> operator()(const Image<HSV>&  img) const;
};

inline Image<BGR> operator|(const Image<Gray>& img, ToBGR op) { return op(img); }
inline Image<BGR> operator|(const Image<HSV>&  img, ToBGR op) { return op(img); }

} // namespace improc::core
