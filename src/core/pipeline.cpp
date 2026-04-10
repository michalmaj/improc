// src/core/pipeline.cpp
#include "improc/core/pipeline.hpp"
#include <utility>

namespace improc::core {

Image<Gray>    ToGray::operator()(Image<BGR>  img) const { return convert<Gray,    BGR> (std::move(img)); }
Image<BGR>     ToBGR::operator()(Image<Gray>  img) const { return convert<BGR,     Gray>(std::move(img)); }
Image<Float32> ToFloat32::operator()(Image<Gray> img) const { return convert<Float32, Gray>(std::move(img)); }

} // namespace improc::core
