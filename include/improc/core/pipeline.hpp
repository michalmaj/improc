// include/improc/core/pipeline.hpp
#pragma once

#include "improc/core/image.hpp"
#include "improc/core/convert.hpp"

namespace improc::core {

template<typename Format, typename Op>
auto operator|(Image<Format> img, Op&& op) {
    return std::forward<Op>(op)(std::move(img));
}

struct ToGray    { Image<Gray>    operator()(Image<BGR>  img) const; };
struct ToBGR     { Image<BGR>     operator()(Image<Gray> img) const; };
struct ToFloat32 { Image<Float32> operator()(Image<Gray> img) const; };

} // namespace improc::core
