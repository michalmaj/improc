// include/improc/core/format_traits.hpp
#pragma once

#include <opencv2/core.hpp>

namespace improc::core {

struct BGR     {};
struct Gray    {};
struct BGRA    {};
struct Float32   {};
struct Float32C3 {};

template<typename Format> struct FormatTraits;  // intentionally undefined — unknown format = compile error

template<> struct FormatTraits<BGR>      { static constexpr int cv_type = CV_8UC3;  static constexpr int channels = 3; };
template<> struct FormatTraits<Gray>     { static constexpr int cv_type = CV_8UC1;  static constexpr int channels = 1; };
template<> struct FormatTraits<BGRA>     { static constexpr int cv_type = CV_8UC4;  static constexpr int channels = 4; };
template<> struct FormatTraits<Float32>  { static constexpr int cv_type = CV_32FC1; static constexpr int channels = 1; };
template<> struct FormatTraits<Float32C3> { static constexpr int cv_type = CV_32FC3; static constexpr int channels = 3; };

} // namespace improc::core
