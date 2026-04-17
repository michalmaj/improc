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

template<AnyFormat Format>
struct Compose : detail::BindMixin<Compose<Format>> {
    using AugFn = std::function<Image<Format>(Image<Format>, std::mt19937&)>;

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
    template<typename Aug>
    RandomApply(Aug aug, float p) : p_(validate_p(p)), aug_(std::move(aug)) {}

    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return img;
        return aug_(std::move(img), rng);
    }
};

template<AnyFormat Format>
struct OneOf : detail::BindMixin<OneOf<Format>> {
    using AugFn = std::function<Image<Format>(Image<Format>, std::mt19937&)>;

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
