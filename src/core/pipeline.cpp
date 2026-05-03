// src/core/pipeline.cpp
#include "improc/core/pipeline.hpp"
#include "improc/core/ops/to_hsv.hpp"
#include "improc/core/ops/to_bgr.hpp"
#include <utility>

namespace improc::core {

Image<Gray>    ToGray::operator()(Image<BGR>  img) const { return convert<Gray,    BGR> (std::move(img)); }
Image<BGR>     ToBGR::operator()(Image<Gray> img) const { return convert<BGR,     Gray>(std::move(img)); }
Image<BGR>     ToBGR::operator()(Image<HSV>  img) const { return convert<BGR,     HSV> (std::move(img)); }
Image<Float32>   ToFloat32::operator()(Image<Gray> img) const   { return convert<Float32,   Gray>(std::move(img)); }
Image<Float32C3> ToFloat32C3::operator()(Image<BGR> img) const { return convert<Float32C3, BGR> (std::move(img)); }
Image<HSV>     ToHSV::operator()(Image<BGR> img) const { return convert<HSV,     BGR> (std::move(img)); }
Image<LAB>     ToLAB::operator()(Image<BGR> img) const { return convert<LAB,     BGR> (std::move(img)); }

} // namespace improc::core
