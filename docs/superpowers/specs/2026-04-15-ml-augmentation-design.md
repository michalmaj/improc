# Design: `improc::ml` — Image Augmentation

**Date:** 2026-04-15
**Status:** Approved

## Overview

Add a stochastic image augmentation module to `improc::ml`. Augmentations live in `include/improc/ml/augment/` and are grouped by category: geometric, color, noise, and composition. All ops are header-only, templated on `Format`, and constrained via C++20 concepts defined in a new `include/improc/core/concepts.hpp`. Each op supports two call styles: `aug(img, rng)` (direct) and `aug.bind(rng)` returning a functor compatible with the `operator|` pipeline.

## Goals

- `RandomFlip`, `RandomRotate`, `RandomCrop`, `RandomResize` — geometric augmentations
- `RandomBrightness`, `RandomContrast`, `ColorJitter` — color/intensity augmentations
- `RandomGaussianNoise`, `RandomSaltAndPepper` — noise augmentations
- `Compose<Format>`, `RandomApply<Format>`, `OneOf<Format>` — composition primitives
- C++20 concepts for format constraints with clear compiler errors
- Reproducible via caller-supplied `std::mt19937&`
- Fluent builder API consistent with existing ops

## Non-Goals

- Retrofitting concepts onto existing `improc::core` ops (separate future task)
- Custom exception hierarchy (separate future task)
- GPU-accelerated augmentation (`improc::cuda`)
- Elastic distortion, grid distortion, optical distortion
- Cutout / CutMix / MixUp (dataset-level augmentations)

---

## File Structure

```
include/improc/core/concepts.hpp                 — AnyFormat, BGRFormat, GrayFormat, MultiChannelFormat

include/improc/ml/augment/detail.hpp             — BindMixin<Derived> (shared by all augmentation headers)
include/improc/ml/augment/geometric.hpp          — RandomFlip, RandomRotate, RandomCrop, RandomResize
include/improc/ml/augment/color.hpp              — RandomBrightness, RandomContrast, ColorJitter
include/improc/ml/augment/noise.hpp              — RandomGaussianNoise, RandomSaltAndPepper
include/improc/ml/augment/compose.hpp            — Compose<Format>, RandomApply<Format>, OneOf<Format>
include/improc/ml/augmentation.hpp               — umbrella include

tests/ml/augment/test_geometric.cpp
tests/ml/augment/test_color.cpp
tests/ml/augment/test_noise.cpp
tests/ml/augment/test_compose.cpp
```

No new `.cpp` files — all implementations are templated and defined in headers.
No `CMakeLists.txt` changes — sources and tests are auto-discovered via `GLOB_RECURSE`.

---

## `concepts.hpp`

```cpp
// include/improc/core/concepts.hpp
#pragma once

#include "improc/core/format_traits.hpp"

namespace improc::core {

template<typename F>
concept AnyFormat = requires {
    { FormatTraits<F>::cv_type }  -> std::convertible_to<int>;
    { FormatTraits<F>::channels } -> std::convertible_to<int>;
};

template<typename F>
concept BGRFormat = AnyFormat<F> && std::same_as<F, BGR>;

template<typename F>
concept GrayFormat = AnyFormat<F> && std::same_as<F, Gray>;

template<typename F>
concept MultiChannelFormat = AnyFormat<F> && (FormatTraits<F>::channels > 1);

} // namespace improc::core
```

---

## `detail.hpp` — Shared `BindMixin<Derived>`

Defined once in `include/improc/ml/augment/detail.hpp`, included by all four augmentation headers. Provides `.bind(rng)` for all augmentations:

```cpp
// include/improc/ml/augment/detail.hpp
#pragma once
#include <random>

namespace improc::ml::detail {
template<typename Derived>
struct BindMixin {
    auto bind(std::mt19937& rng) const {
        return [derived = static_cast<const Derived&>(*this), &rng](auto img) {
            return derived(std::move(img), rng);
        };
    }
};
} // namespace improc::ml::detail
```

---

## `geometric.hpp`

```cpp
// include/improc/ml/augment/geometric.hpp
#pragma once

#include <random>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/core/ops/axis.hpp"
#include "improc/ml/augment/detail.hpp"

namespace improc::ml {

struct RandomFlip : detail::BindMixin<RandomFlip> {
    RandomFlip& p(float prob);           // [0,1], default 0.5; throws std::invalid_argument otherwise
    RandomFlip& axis(core::Axis a);      // Horizontal/Vertical/Both, default Horizontal

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // std::bernoulli_distribution(p_), cv::flip

private:
    float      p_    = 0.5f;
    core::Axis axis_ = core::Axis::Horizontal;
};

struct RandomRotate : detail::BindMixin<RandomRotate> {
    RandomRotate& range(float min_deg, float max_deg);  // default [-15, 15]; throws if min > max
    RandomRotate& scale(float s);                        // default 1.0; throws if s <= 0

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // std::uniform_real_distribution, cv::getRotationMatrix2D + cv::warpAffine

private:
    float min_deg_ = -15.0f;
    float max_deg_ =  15.0f;
    float scale_   =   1.0f;
};

struct RandomCrop : detail::BindMixin<RandomCrop> {
    RandomCrop& width(int w);   // required; throws if w <= 0
    RandomCrop& height(int h);  // required; throws if h <= 0

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // throws std::invalid_argument if crop size > image size
    // std::uniform_int_distribution for x offset, y offset

private:
    int width_  = 0;
    int height_ = 0;
};

struct RandomResize : detail::BindMixin<RandomResize> {
    RandomResize& range(int min_side, int max_side);  // shorter side target; throws if min > max or min <= 0

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // samples target shorter side, preserves aspect ratio, cv::resize

private:
    int min_side_ = 224;
    int max_side_ = 256;
};

} // namespace improc::ml
```

---

## `color.hpp`

```cpp
// include/improc/ml/augment/color.hpp
#pragma once

#include <random>
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::ml {

struct RandomBrightness : detail::BindMixin<RandomBrightness> {
    RandomBrightness& range(float low, float high);  // multiplicative factor; default [0.8, 1.2]
    // throws if low <= 0 or low > high

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // converts to CV_32F, multiplies by factor, clamps to [0,255], converts back

private:
    float low_  = 0.8f;
    float high_ = 1.2f;
};

struct RandomContrast : detail::BindMixin<RandomContrast> {
    RandomContrast& range(float low, float high);  // alpha ∈ [low,high]; default [0.8, 1.2]
    // throws if low <= 0 or low > high
    // output = alpha * img + (1 - alpha) * mean

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // cv::meanStdDev for mean, img.convertTo with alpha/beta

private:
    float low_  = 0.8f;
    float high_ = 1.2f;
};

struct ColorJitter : detail::BindMixin<ColorJitter> {
    ColorJitter& brightness(float low, float high);  // default [0.8, 1.2]; throws if low <= 0 or low > high
    ColorJitter& contrast(float low, float high);    // default [0.8, 1.2]; throws if low <= 0 or low > high
    ColorJitter& saturation(float low, float high);  // default [0.8, 1.2]; throws if low <= 0 or low > high
    ColorJitter& hue(float low, float high);         // delta degrees; default [-10, 10]; throws if |low|>180 or low>high

    template<BGRFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // BGR→HSV, jitter H (add delta), S (multiply), V (multiply for brightness/contrast), HSV→BGR

private:
    float br_low_ = 0.8f,  br_high_ = 1.2f;
    float ct_low_ = 0.8f,  ct_high_ = 1.2f;
    float sa_low_ = 0.8f,  sa_high_ = 1.2f;
    float hu_low_ = -10.f, hu_high_ = 10.0f;
};

} // namespace improc::ml
```

**Note on `ColorJitter`:** `template<BGRFormat Format>` — using non-BGR triggers a concept constraint violation at compile time with a clear error (`BGRFormat` not satisfied), not a runtime error.

---

## `noise.hpp`

```cpp
// include/improc/ml/augment/noise.hpp
#pragma once

#include <random>
#include <stdexcept>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::ml {

struct RandomGaussianNoise : detail::BindMixin<RandomGaussianNoise> {
    RandomGaussianNoise& std_dev(float low, float high);  // default [5.0, 20.0]; throws if low < 0 or low > high
    RandomGaussianNoise& mean(float m);                   // default 0.0

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // cv::randn noise mat (CV_32F), add to img (cast to float), clamp, cast back

private:
    float std_low_  =  5.0f;
    float std_high_ = 20.0f;
    float mean_     =  0.0f;
};

struct RandomSaltAndPepper : detail::BindMixin<RandomSaltAndPepper> {
    RandomSaltAndPepper& p(float prob);  // fraction of noisy pixels; default 0.05; throws if p < 0 or p > 1

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // std::bernoulli_distribution(p_) per pixel
    // std::bernoulli_distribution(0.5) decides salt (255) vs pepper (0)
    // multi-channel: sets all channels of chosen pixel

private:
    float p_ = 0.05f;
};

} // namespace improc::ml
```

---

## `compose.hpp`

```cpp
// include/improc/ml/augment/compose.hpp
#pragma once

#include <functional>
#include <random>
#include <stdexcept>
#include <vector>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::ml {

template<core::AnyFormat Format>
struct Compose : detail::BindMixin<Compose<Format>> {
    using AugFn = std::function<Image<Format>(Image<Format>, std::mt19937&)>;

    template<typename Aug>
    Compose& add(Aug aug) { steps_.emplace_back(std::move(aug)); return *this; }

    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // applies all steps sequentially

private:
    std::vector<AugFn> steps_;
};

template<core::AnyFormat Format>
struct RandomApply : detail::BindMixin<RandomApply<Format>> {
    using AugFn = std::function<Image<Format>(Image<Format>, std::mt19937&)>;

    template<typename Aug>
    RandomApply(Aug aug, float p);
    // throws std::invalid_argument if p < 0 or p > 1

    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // std::bernoulli_distribution(p_); if false returns img unchanged

private:
    AugFn aug_;
    float p_ = 0.5f;
};

template<core::AnyFormat Format>
struct OneOf : detail::BindMixin<OneOf<Format>> {
    using AugFn = std::function<Image<Format>(Image<Format>, std::mt19937&)>;

    template<typename Aug>
    OneOf& add(Aug aug) { options_.emplace_back(std::move(aug)); return *this; }

    Image<Format> operator()(Image<Format> img, std::mt19937& rng) const;
    // throws std::logic_error if options_ is empty
    // std::uniform_int_distribution(0, options_.size()-1)

private:
    std::vector<AugFn> options_;
};

} // namespace improc::ml
```

---

## `augmentation.hpp` (umbrella)

```cpp
// include/improc/ml/augmentation.hpp
#pragma once

#include "improc/ml/augment/geometric.hpp"
#include "improc/ml/augment/color.hpp"
#include "improc/ml/augment/noise.hpp"
#include "improc/ml/augment/compose.hpp"
```

---

## Error Handling Summary

| Situation | Exception |
|---|---|
| `p` outside `[0, 1]` | `std::invalid_argument` |
| `range(low, high)` where `low > high` | `std::invalid_argument` |
| `low <= 0` for brightness/contrast | `std::invalid_argument` |
| `RandomCrop` larger than image | `std::invalid_argument` |
| `min_side <= 0` or `min > max` for RandomResize | `std::invalid_argument` |
| `scale <= 0` for RandomRotate | `std::invalid_argument` |
| `OneOf` with empty options list | `std::logic_error` |
| `ColorJitter` on non-BGR format | compile error (concept `BGRFormat`) |
| OpenCV error inside op | `std::runtime_error` (wrapped `cv::Exception`) |

---

## Tests

### `test_geometric.cpp`
- `RandomFlip` preserves size/type; output differs from input at p=1.0
- `RandomFlip` with p=0.0 returns unchanged image (fixed seed)
- `RandomFlip` invalid p throws `std::invalid_argument`
- `RandomRotate` preserves size/type
- `RandomRotate` min > max throws `std::invalid_argument`
- `RandomRotate` scale <= 0 throws `std::invalid_argument`
- `RandomCrop` returns correct output size
- `RandomCrop` larger than image throws `std::invalid_argument`
- `RandomCrop` missing width/height throws `std::invalid_argument`
- `RandomResize` output shorter side within range
- `RandomResize` min > max throws `std::invalid_argument`
- Pipeline ops via `.bind(rng)` for all four

### `test_color.cpp`
- `RandomBrightness` preserves size/type; output differs from input
- `RandomBrightness` low <= 0 throws
- `RandomBrightness` low > high throws
- `RandomContrast` preserves size/type
- `RandomContrast` invalid range throws
- `ColorJitter` on BGR preserves size/type; output differs
- `ColorJitter` on Gray — compile-time constraint only (concept violation); verified via a `static_assert(!BGRFormat<Gray>)` sanity check, not a runtime test
- All setter validations

### `test_noise.cpp`
- `RandomGaussianNoise` preserves size/type; output differs from input
- `RandomGaussianNoise` std_dev low < 0 throws
- `RandomSaltAndPepper` preserves size/type
- `RandomSaltAndPepper` p=1.0 — all pixels are 0 or 255
- `RandomSaltAndPepper` p < 0 throws
- `RandomSaltAndPepper` p > 1 throws
- Pipeline ops via `.bind(rng)` for both

### `test_compose.cpp`
- `Compose` applies all steps in order (fixed seed, verify cumulative effect)
- `Compose` with zero steps returns unchanged image
- `RandomApply` with p=1.0 always applies; p=0.0 never applies
- `RandomApply` invalid p throws
- `OneOf` with single option always applies that option
- `OneOf` with empty list throws `std::logic_error`
- `OneOf` distributes across options (fixed seed, verify variety over N calls)
- Nested `Compose` inside `Compose`
- `.bind(rng)` for all three with `operator|`

---

## Usage Examples

```cpp
#include "improc/ml/augmentation.hpp"
using namespace improc::ml;
using namespace improc::core;

std::mt19937 rng(42);

// Standalone call
Image<BGR> aug = RandomFlip{}.p(0.5f)(img, rng);

// Pipeline via .bind()
Image<BGR> result = img
    | RandomFlip{}.p(0.5f).bind(rng)
    | RandomBrightness{}.range(0.8f, 1.2f).bind(rng);

// Composition pipeline for training
auto augmentor = Compose<BGR>{}
    .add(RandomFlip{}.p(0.5f))
    .add(RandomRotate{}.range(-10.0f, 10.0f))
    .add(RandomApply<BGR>{ColorJitter{}.brightness(0.8f, 1.2f).saturation(0.9f, 1.1f), 0.5f})
    .add(OneOf<BGR>{}
        .add(RandomGaussianNoise{}.std_dev(5.0f, 15.0f))
        .add(RandomSaltAndPepper{}.p(0.02f)));

for (auto& img : dataset) {
    Image<BGR> augmented = augmentor(img, rng);
    // feed to training
}

// Letterbox + augment for detection training
Image<BGR> ready = img
    | Resize{}.width(640)
    | PadToSquare{}
    | augmentor.bind(rng);
```
