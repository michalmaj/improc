// include/improc/ml/augment/seg_compose.hpp
#pragma once

#include <functional>
#include <vector>
#include "improc/ml/segmented.hpp"
#include "improc/ml/augment/detail.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

template<AnyFormat Format>
struct SegCompose : detail::BindMixin<SegCompose<Format>> {
    using Op = std::function<SegmentedImage<Format>(SegmentedImage<Format>, std::mt19937&)>;

    SegCompose& add(Op op) {
        if (!op) throw ParameterError{"op", "must not be null", "SegCompose"};
        ops_.push_back(std::move(op));
        return *this;
    }

    SegmentedImage<Format> operator()(SegmentedImage<Format> seg, std::mt19937& rng) const {
        for (const auto& op : ops_)
            seg = op(std::move(seg), rng);
        return seg;
    }

private:
    std::vector<Op> ops_;
};

} // namespace improc::ml
