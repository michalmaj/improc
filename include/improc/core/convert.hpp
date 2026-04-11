// include/improc/core/convert.hpp
#pragma once

#include "improc/core/image.hpp"

namespace improc::core {

// Primary template: deleted — unknown conversion = compile error
template<typename To, typename From>
Image<To> convert(const Image<From>&) = delete;

// Allowed conversions (full explicit specializations)
template<> Image<Gray>    convert<Gray,    BGR> (const Image<BGR>&     src);
template<> Image<BGR>     convert<BGR,     Gray>(const Image<Gray>&    src);
template<> Image<BGRA>    convert<BGRA,    BGR> (const Image<BGR>&     src);
template<> Image<BGR>     convert<BGR,     BGRA>(const Image<BGRA>&    src);
template<> Image<Float32>   convert<Float32,   Gray>(const Image<Gray>&    src);
template<> Image<Float32C3> convert<Float32C3, BGR> (const Image<BGR>&     src);

} // namespace improc::core
