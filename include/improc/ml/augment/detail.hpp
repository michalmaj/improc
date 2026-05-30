// include/improc/ml/augment/detail.hpp
#pragma once

#include <random>

namespace improc::ml::detail {

/**
 * @brief CRTP mixin that adds `bind(rng)` to augmentation ops.
 *
 * Inherit from `BindMixin<Derived>` to get a `bind()` method that wraps
 * `operator()(img, rng)` into a unary functor compatible with `operator|`.
 *
 * @code
 * std::mt19937 rng(42);
 * img = img | RandomFlip().bind(rng);
 * @endcode
 *
 * @tparam Derived The concrete augmentation struct.
 */
template<typename Derived>
struct BindMixin {
    /// @brief Returns a unary functor `Image<F> → Image<F>` for use with `operator|`.
    /// @warning The caller's `rng` must outlive the returned functor.
    [[nodiscard]] auto bind(std::mt19937& rng) const {
        return [derived = static_cast<const Derived&>(*this), &rng](auto img) {
            return derived(std::move(img), rng);
        };
    }
};

} // namespace improc::ml::detail
