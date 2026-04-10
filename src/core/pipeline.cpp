// src/core/pipeline.cpp
#include "improc/core/pipeline.hpp"

namespace improc::core {

Image<Gray>    ToGray::operator()(Image<BGR>  img) const { return convert<Gray,    BGR> (img); }
Image<BGR>     ToBGR::operator()(Image<Gray>  img) const { return convert<BGR,     Gray>(img); }
Image<Float32> ToFloat32::operator()(Image<Gray> img) const { return convert<Float32, Gray>(img); }

} // namespace improc::core
