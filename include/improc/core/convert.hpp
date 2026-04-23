// include/improc/core/convert.hpp
#pragma once

#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Converts between image formats at compile time.
 *
 * Only explicit specializations compile; the primary template is deleted.
 *
 * @tparam To    Target format tag.
 * @tparam From  Source format tag.
 * @param  src   Source image.
 * @return       New `Image<To>` containing the converted data.
 *
 * @code
 * Image<Gray> gray = convert<Gray>(bgr_img);
 * Image<HSV>  hsv  = convert<HSV>(bgr_img);
 * @endcode
 */
template<typename To, typename From>
Image<To> convert(const Image<From>&) = delete;

/// Supported conversions (explicit specializations only):
template<> Image<Gray>    convert<Gray,    BGR> (const Image<BGR>&     src);
template<> Image<BGR>     convert<BGR,     Gray>(const Image<Gray>&    src);
template<> Image<BGRA>    convert<BGRA,    BGR> (const Image<BGR>&     src);
template<> Image<BGR>     convert<BGR,     BGRA>(const Image<BGRA>&    src);
template<> Image<Float32>   convert<Float32,   Gray>     (const Image<Gray>&      src);
template<> Image<Float32C3> convert<Float32C3, BGR>      (const Image<BGR>&       src);
template<> Image<Gray>      convert<Gray,      Float32>  (const Image<Float32>&   src);
template<> Image<BGR>       convert<BGR,       Float32C3>(const Image<Float32C3>& src);
template<> Image<HSV> convert<HSV, BGR>(const Image<BGR>& src);
template<> Image<BGR> convert<BGR, HSV>(const Image<HSV>& src);

} // namespace improc::core
