// include/improc/ml/augment/compose.hpp
#pragma once

#include <functional>
#include <random>
#include <stdexcept>
#include <vector>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/ml/augment/detail.hpp"

namespace improc::ml {

using improc::core::Image;

template<improc::core::AnyFormat Format>
struct Compose : detail::BindMixin<Compose<Format>> {
    using AugFn = std::function<Image<Format>(Image<Format>, std::mt19937&)>;

    template<typename Aug>
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

template<improc::core::AnyFormat Format>
struct RandomApply : detail::BindMixin<RandomApply<Format>> {
    using AugFn = std::function<Image<Format>(Image<Format>, std::mt19937&)>;

    template<typename Aug>
    RandomApply(Aug aug, float p) : aug_(std::move(aug)), p_(p) {
        if (p < 0.0f || p > 1.0f)
            throw std::invalid_argument("RandomApply: p must be in [0, 1]");
    }

    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        std::bernoulli_distribution d(p_);
        if (!d(rng)) return img;
        return aug_(std::move(img), rng);
    }

private:
    AugFn aug_;
    float p_ = 0.5f;
};

template<improc::core::AnyFormat Format>
struct OneOf : detail::BindMixin<OneOf<Format>> {
    using AugFn = std::function<Image<Format>(Image<Format>, std::mt19937&)>;

    template<typename Aug>
    OneOf& add(Aug aug) {
        options_.emplace_back(std::move(aug));
        return *this;
    }

    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const {
        if (options_.empty())
            throw std::logic_error("OneOf: no augmentations added");
        std::uniform_int_distribution<std::size_t> d(0, options_.size() - 1);
        return options_[d(rng)](std::move(img), rng);
    }

private:
    std::vector<AugFn> options_;
};

} // namespace improc::ml
