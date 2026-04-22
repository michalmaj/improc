// include/improc/core/pipeline.hpp
#pragma once

#include "improc/core/image.hpp"
#include "improc/core/convert.hpp"
#include "improc/core/ops/axis.hpp"
#include "improc/core/ops/resize.hpp"
#include "improc/core/ops/crop.hpp"
#include "improc/core/ops/flip.hpp"
#include "improc/core/ops/rotate.hpp"
#include "improc/core/ops/normalize.hpp"
#include "improc/core/ops/blur.hpp"
#include "improc/core/ops/morphology.hpp"
#include "improc/core/ops/threshold.hpp"
#include "improc/core/ops/pad.hpp"
#include "improc/core/ops/clahe.hpp"
#include "improc/core/ops/gamma.hpp"
#include "improc/core/ops/bilateral_filter.hpp"
#include "improc/core/ops/edge.hpp"
#include "improc/core/ops/homography.hpp"
#include "improc/core/ops/warp_affine.hpp"
#include "improc/core/ops/apply_mask.hpp"
#include "improc/core/ops/unsharp_mask.hpp"
#include "improc/core/ops/to_hsv.hpp"
#include "improc/core/ops/to_bgr.hpp"

namespace improc::core {

template<AnyFormat Format, typename Op>
auto operator|(Image<Format> img, Op&& op) {
    return std::forward<Op>(op)(std::move(img));
}

struct ToGray      { Image<Gray>      operator()(Image<BGR>  img) const; };
struct ToFloat32   { Image<Float32>   operator()(Image<Gray> img) const; };
struct ToFloat32C3 { Image<Float32C3> operator()(Image<BGR>  img) const; };

} // namespace improc::core
