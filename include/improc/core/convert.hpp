// include/improc/core/convert.hpp
#pragma once

#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Converts an image from one format to another at compile time.
 *
 * The primary template is deleted — calling `convert<To, From>` for an
 * unsupported pair is a compile error. Only the explicit specializations
 * listed below are valid.
 *
 * @tparam To    Target format tag.
 * @tparam From  Source format tag.
 * @return       New `Image<To>` containing the converted data.
 *
 * @code
 * Image<Gray> gray = convert<Gray>(bgr_img);   // OK — specialization exists
 * // Image<HSV> bad = convert<HSV>(gray_img);  // compile error — not supported
 * @endcode
 */
template<typename To, typename From>
Image<To> convert(const Image<From>&) = delete;

/// @brief BGR → Gray (luminosity).
template<> Image<Gray>      convert<Gray,      BGR>      (const Image<BGR>&       src);
/// @brief Gray → BGR (replicated channels).
template<> Image<BGR>       convert<BGR,       Gray>     (const Image<Gray>&      src);
/// @brief BGR → BGRA (alpha channel set to 255).
template<> Image<BGRA>      convert<BGRA,      BGR>      (const Image<BGR>&       src);
/// @brief BGRA → BGR (alpha channel dropped).
template<> Image<BGR>       convert<BGR,       BGRA>     (const Image<BGRA>&      src);
/// @brief Gray → Float32 (values scaled to [0, 1]).
template<> Image<Float32>   convert<Float32,   Gray>     (const Image<Gray>&      src);
/// @brief BGR → Float32C3 (values scaled to [0, 1]).
template<> Image<Float32C3> convert<Float32C3, BGR>      (const Image<BGR>&       src);
/// @brief Float32 → Gray (values scaled to [0, 255]).
template<> Image<Gray>      convert<Gray,      Float32>  (const Image<Float32>&   src);
/// @brief Float32C3 → BGR (values scaled to [0, 255]).
template<> Image<BGR>       convert<BGR,       Float32C3>(const Image<Float32C3>& src);
/// @brief BGR → HSV.
template<> Image<HSV>       convert<HSV,       BGR>      (const Image<BGR>&       src);
/// @brief HSV → BGR.
template<> Image<BGR>       convert<BGR,       HSV>      (const Image<HSV>&       src);
/// @brief BGR → CIE L*a*b*.
template<> Image<LAB>       convert<LAB,       BGR>      (const Image<BGR>&       src);
/// @brief LAB → BGR.
template<> Image<BGR>       convert<BGR,       LAB>      (const Image<LAB>&       src);

} // namespace improc::core
