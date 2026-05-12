// include/improc/ml/labeled.hpp
#pragma once

#include <utility>
#include <vector>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::ml {

using improc::core::AnyFormat;
using improc::core::Image;

template<AnyFormat Format>
struct LabeledImage {
    Image<Format>       image;
    std::vector<float>  label;
};

template<AnyFormat Format, typename Op>
auto operator|(LabeledImage<Format> li, Op&& op) {
    return std::forward<Op>(op)(std::move(li));
}

} // namespace improc::ml
