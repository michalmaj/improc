// include/improc/core/format_traits.hpp
#pragma once

#include <opencv2/core.hpp>
#include <string_view>

namespace improc::core {

/// @brief 8-bit BGR color image. 3 channels, CV_8UC3.
struct BGR     {};
/// @brief 8-bit single-channel grayscale image. CV_8UC1.
struct Gray    {};
/// @brief 8-bit BGR + alpha image. 4 channels, CV_8UC4.
struct BGRA    {};
/// @brief Single-channel 32-bit float image. CV_32FC1. Values typically in [0, 1].
struct Float32   {};
/// @brief 3-channel 32-bit float image. CV_32FC3. Values typically in [0, 1].
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

template<> struct FormatTraits<BGR>       { static constexpr int cv_type = CV_8UC3;  static constexpr int channels = 3; static constexpr bool is_float = false; static constexpr std::string_view name = "BGR (CV_8UC3)";       };
template<> struct FormatTraits<Gray>      { static constexpr int cv_type = CV_8UC1;  static constexpr int channels = 1; static constexpr bool is_float = false; static constexpr std::string_view name = "Gray (CV_8UC1)";      };
template<> struct FormatTraits<BGRA>      { static constexpr int cv_type = CV_8UC4;  static constexpr int channels = 4; static constexpr bool is_float = false; static constexpr std::string_view name = "BGRA (CV_8UC4)";      };
template<> struct FormatTraits<Float32>   { static constexpr int cv_type = CV_32FC1; static constexpr int channels = 1; static constexpr bool is_float = true;  static constexpr std::string_view name = "Float32 (CV_32FC1)";  };
template<> struct FormatTraits<Float32C3> { static constexpr int cv_type = CV_32FC3; static constexpr int channels = 3; static constexpr bool is_float = true;  static constexpr std::string_view name = "Float32C3 (CV_32FC3)"; };

/// @brief HSV color space. 8-bit, 3 channels. H∈[0,179], S∈[0,255], V∈[0,255].
struct HSV {};
template<> struct FormatTraits<HSV> {
    static constexpr int cv_type  = CV_8UC3;
    static constexpr int channels = 3;
    static constexpr bool is_float = false;
    static constexpr std::string_view name = "HSV (CV_8UC3)";
};

} // namespace improc::core
