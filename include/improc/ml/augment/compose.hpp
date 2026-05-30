// include/improc/ml/augment/compose.hpp
#pragma once

#include <concepts>
#include <functional>
#include <random>
#include <vector>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/ml/augment/detail.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

using improc::core::AnyFormat;
using improc::core::Image;

/**
 * @brief Sequential pipeline of augmentation ops applied to `Image<Format>`.
 *
 * Each op is stored as `std::function<Image<Format>(Image<Format>, std::mt19937&)>`.
 * Ops are applied in insertion order. Inherits `bind(rng)` from `BindMixin`.
 *
 * @code
 * std::mt19937 rng(42);
 * Compose<BGR> aug;
 * aug.add([](Image<BGR> img, std::mt19937& r) { return RandomFlip{}(img, r); })
 *    .add([](Image<BGR> img, std::mt19937& r) { return ColorJitter{}(img, r); });
 * img = img | aug.bind(rng);
 * @endcode
 */
template<AnyFormat Format>
struct Compose : detail::BindMixin<Compose<Format>> {
    using AugFn = std::function<Image<Format>(Image<Format>, std::mt19937&)>;

    /// @brief Appends an augmentation op to the pipeline.
    template<typename Aug>
        requires std::invocable<Aug&, Image<Format>, std::mt19937&>
    Compose& add(Aug aug) {
        steps_.emplace_back(std::move(aug));
        return *this;
    }

    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        for (const auto& step : steps_)
            img = step(std::move(img), rng);
        return img;
    }

private:
    std::vector<AugFn> steps_;
};

/**
 * @brief Applies a single augmentation op with probability `p`.
 *
 * Wraps any op that takes `(Image<Format>, std::mt19937&)`.
 */
template<AnyFormat Format>
struct RandomApply : detail::BindMixin<RandomApply<Format>> {
    using AugFn = std::function<Image<Format>(Image<Format>, std::mt19937&)>;

private:
    // p_ declared before aug_ so the member-initialiser validates p before aug is moved
    float p_;
    AugFn aug_;

    static float validate_p(float p) {
        if (p < 0.0f || p > 1.0f)
            throw ParameterError{"p", std::format("must be in [0, 1], got {}", p), "RandomApply"};
        return p;
    }

public:
    /// @brief Constructs with an augmentation op and application probability.
    /// @throws improc::ParameterError if `prob` is outside [0, 1].
    template<typename Aug>
    RandomApply(Aug aug, float p) : p_(validate_p(p)), aug_(std::move(aug)) {}

    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return img;
        return aug_(std::move(img), rng);
    }
};

/**
 * @brief Uniformly samples one op from a registered pool and applies it.
 */
template<AnyFormat Format>
struct OneOf : detail::BindMixin<OneOf<Format>> {
    using AugFn = std::function<Image<Format>(Image<Format>, std::mt19937&)>;

    /// @brief Adds an op to the selection pool.
    template<typename Aug>
        requires std::invocable<Aug&, Image<Format>, std::mt19937&>
    OneOf& add(Aug aug) {
        options_.emplace_back(std::move(aug));
        return *this;
    }

    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        if (options_.empty())
            throw AugmentError{"OneOf: operator() called with no augmentations added"};
        std::uniform_int_distribution<std::size_t> d(0, options_.size() - 1);
        return options_[d(rng)](std::move(img), rng);
    }

private:
    std::vector<AugFn> options_;
};

} // namespace improc::ml
