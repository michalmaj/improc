// include/improc/ml/augment/bbox_compose.hpp
#pragma once

#include <functional>
#include <vector>
#include "improc/ml/annotated.hpp"
#include "improc/ml/augment/detail.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

/**
 * @brief Sequential pipeline of augmentation ops applied to `AnnotatedImage<Format>` (image + bounding boxes).
 *
 * Each op receives and returns the full annotated image, allowing ops to update box coordinates.
 * Inherits `bind(rng)` from `BindMixin`.
 */
template<AnyFormat Format>
struct BBoxCompose : detail::BindMixin<BBoxCompose<Format>> {
    using Op = std::function<AnnotatedImage<Format>(AnnotatedImage<Format>, std::mt19937&)>;

    /// @brief Appends an annotated-image augmentation op.
    /// @throws improc::ParameterError if `op` is null.
    BBoxCompose& add(Op op) {
        if (!op) throw ParameterError{"op", "must not be null", "BBoxCompose"};
        ops_.push_back(std::move(op));
        return *this;
    }

    [[nodiscard]] AnnotatedImage<Format> operator()(AnnotatedImage<Format> ann, std::mt19937& rng) const {
        for (const auto& op : ops_)
            ann = op(std::move(ann), rng);
        return ann;
    }

private:
    std::vector<Op> ops_;
};

} // namespace improc::ml
