// include/improc/core/format_traits.hpp
#pragma once

#include <opencv2/core.hpp>
#include <string_view>

namespace improc::core {

/**
 * @brief 8-bit BGR color image. 3 channels, CV_8UC3.
 * @code Image<BGR> img(mat); @endcode
 */
struct BGR     {};

/**
 * @brief 8-bit single-channel grayscale image. CV_8UC1.
 * @code Image<Gray> g(mat); @endcode
 */
struct Gray    {};

/**
 * @brief 8-bit BGR + alpha image. 4 channels, CV_8UC4.
 * @code Image<BGRA> img(mat); @endcode
 */
struct BGRA    {};

/**
 * @brief Single-channel 32-bit float image. CV_32FC1. Values typically in [0, 1].
 * @code Image<Float32> f(mat); @endcode
 */
struct Float32   {};

/**
 * @brief 3-channel 32-bit float image. CV_32FC3. Values typically in [0, 1].
 * @code Image<Float32C3> f(mat); @endcode
 */
struct Float32C3 {};

/**
 * @brief Maps a format tag to its OpenCV type constants.
 *
 * Specializations define `cv_type`, `channels`, `is_float`, and `name`.
 * Using an unspecialized `FormatTraits<F>` is a compile error.
 *
 * @tparam Format  A format tag (BGR, Gray, BGRA, HSV, Float32, Float32C3).
 */
template<typename Format> struct FormatTraits;  // intentionally undefined — unknown format = compile error

/// @name FormatTraits specializations
/// @{

/// @brief Traits for BGR: CV_8UC3, 3 channels, integer.
template<> struct FormatTraits<BGR>       { static constexpr int cv_type = CV_8UC3;  static constexpr int channels = 3; static constexpr bool is_float = false; static constexpr std::string_view name = "BGR (CV_8UC3)";       };
/// @brief Traits for Gray: CV_8UC1, 1 channel, integer.
template<> struct FormatTraits<Gray>      { static constexpr int cv_type = CV_8UC1;  static constexpr int channels = 1; static constexpr bool is_float = false; static constexpr std::string_view name = "Gray (CV_8UC1)";      };
/// @brief Traits for BGRA: CV_8UC4, 4 channels, integer.
template<> struct FormatTraits<BGRA>      { static constexpr int cv_type = CV_8UC4;  static constexpr int channels = 4; static constexpr bool is_float = false; static constexpr std::string_view name = "BGRA (CV_8UC4)";      };
/// @brief Traits for Float32: CV_32FC1, 1 channel, float.
template<> struct FormatTraits<Float32>   { static constexpr int cv_type = CV_32FC1; static constexpr int channels = 1; static constexpr bool is_float = true;  static constexpr std::string_view name = "Float32 (CV_32FC1)";  };
/// @brief Traits for Float32C3: CV_32FC3, 3 channels, float.
template<> struct FormatTraits<Float32C3> { static constexpr int cv_type = CV_32FC3; static constexpr int channels = 3; static constexpr bool is_float = true;  static constexpr std::string_view name = "Float32C3 (CV_32FC3)"; };

/**
 * @brief HSV color space. 8-bit, 3 channels. H∈[0,179], S∈[0,255], V∈[0,255].
 * @code Image<HSV> hsv = bgr | ToHSV{}; @endcode
 */
struct HSV {};
/// @brief Traits for HSV: CV_8UC3, 3 channels, integer.
template<> struct FormatTraits<HSV> {
    static constexpr int cv_type  = CV_8UC3;
    static constexpr int channels = 3;
    static constexpr bool is_float = false;
    static constexpr std::string_view name = "HSV (CV_8UC3)";
};

/**
 * @brief CIE L*a*b* color space. 8-bit, 3 channels.
 * OpenCV 8-bit encoding: L ∈ [0, 255], a ∈ [0, 255], b ∈ [0, 255]
 * (actual L* scaled by 255/100; a* and b* shifted by +128).
 * @code Image<LAB> lab = bgr | ToLAB{}; @endcode
 */
struct LAB {};
/// @brief Traits for LAB: CV_8UC3, 3 channels, integer.
template<> struct FormatTraits<LAB> {
    static constexpr int  cv_type  = CV_8UC3;
    static constexpr int  channels = 3;
    static constexpr bool is_float = false;
    static constexpr std::string_view name = "LAB (CV_8UC3)";
};

/**
 * @brief YCbCr color space (OpenCV convention: Y, Cr, Cb channel order). 8-bit, 3 channels.
 * @code Image<YCrCb> y = bgr | ToYCrCb{}; @endcode
 */
struct YCrCb {};
/// @brief Traits for YCrCb: CV_8UC3, 3 channels, integer.
template<> struct FormatTraits<YCrCb> {
    static constexpr int  cv_type  = CV_8UC3;
    static constexpr int  channels = 3;
    static constexpr bool is_float = false;
    static constexpr std::string_view name = "YCrCb (CV_8UC3)";
};

/// @}

} // namespace improc::core
