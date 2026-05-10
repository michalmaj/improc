// include/improc/ml/annotated.hpp
#pragma once

#include <string>
#include <utility>
#include <vector>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::ml {

using improc::core::AnyFormat;
using improc::core::Image;

struct BBox {
    cv::Rect2f  box;
    int         class_id = 0;
    std::string label;
};

template<AnyFormat Format>
struct AnnotatedImage {
    Image<Format>     image;
    std::vector<BBox> boxes;
};

template<AnyFormat Format, typename Op>
auto operator|(AnnotatedImage<Format> ann, Op&& op) {
    return std::forward<Op>(op)(std::move(ann));
}

} // namespace improc::ml
