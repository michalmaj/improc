// include/improc/ml/segmented.hpp
#pragma once

#include <optional>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::ml {

using improc::core::AnyFormat;
using improc::core::Image;
using improc::core::Gray;

template<AnyFormat Format>
struct SegmentedImage {
    Image<Format>              image;
    Image<Gray>                class_mask;    // pixel = class_id; 255 = void (kept as-is)
    std::optional<Image<Gray>> instance_mask; // pixel = instance_id; nullopt when not loaded
};

template<AnyFormat Format, typename Op>
auto operator|(SegmentedImage<Format> seg, Op&& op) {
    return std::forward<Op>(op)(std::move(seg));
}

} // namespace improc::ml
