#pragma once

#include <random>

namespace improc::ml::detail {

template<typename Derived>
struct BindMixin {
    // Returns a unary functor (Image<Format> → Image<Format>) for use with operator|.
    // The caller's rng must outlive the returned functor.
    auto bind(std::mt19937& rng) const {
        return [derived = static_cast<const Derived&>(*this), &rng](auto img) {
            return derived(std::move(img), rng);
        };
    }
};

} // namespace improc::ml::detail
