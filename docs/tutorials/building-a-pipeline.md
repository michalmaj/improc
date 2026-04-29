# Building a Pipeline

This tutorial covers the core pipeline mechanics: composing ops, converting formats, using lazy views for collections, and plugging in augmentation.

## Eager vs Lazy

improc++ has two pipeline styles.

**Eager** (`operator|` on `Image<F>`) — each step executes immediately:

```cpp
#include "improc/core/pipeline.hpp"
using namespace improc::core;

Image<BGR> result = img
    | Resize{}.width(224).height(224)   // executes now → new Image<BGR>
    | GaussianBlur{}.kernel_size(3)     // executes now → new Image<BGR>
    | Brightness{}.delta(10.0);         // executes now → new Image<BGR>
```

**Lazy** (`improc::views`) — builds a deferred chain; nothing runs until you materialise:

```cpp
#include "improc/views/views.hpp"
namespace views = improc::views;

auto view = img
    | views::transform(Resize{}.width(224).height(224))
    | views::transform(GaussianBlur{}.kernel_size(3));

Image<BGR> result = view | views::to<Image<BGR>>();  // executes here, once
```

Use eager pipelines when you have a single image and want readable, step-by-step processing. Use lazy views when you have a collection and want to avoid allocating intermediate vectors.

## Format Conversions

Format mismatches are compiler errors. Use `convert<To>()` or the explicit functor ops to cross format boundaries:

```cpp
// Free function — explicit conversion
Image<Gray>    gray    = convert<Gray>(bgr_img);
Image<Float32C3> f32c3 = convert<Float32C3>(bgr_img);

// In a pipeline — same thing, composable
Image<Float32C3> tensor = img
    | Resize{}.width(224).height(224)
    | ToFloat32C3{}
    | NormalizeTo{0.0f, 1.0f};
```

Available conversions:

| From | To | Function |
|---|---|---|
| `BGR` | `Gray` | `convert<Gray>(img)` |
| `BGR` | `Float32C3` | `convert<Float32C3>(img)` |
| `BGR` | `HSV` | `convert<HSV>(img)` |
| `Gray` | `BGR` | `convert<BGR>(img)` |
| `Float32C3` | `BGR` | `convert<BGR>(img)` |
| `HSV` | `BGR` | `convert<BGR>(img)` |

## All Core Ops

### Geometric

```cpp
img | Resize{}.width(224).height(224)          // exact size
img | Resize{}.width(224)                      // aspect-ratio preserved
img | Crop{}.x(10).y(10).width(100).height(100)
img | Flip{}.axis(Axis::Horizontal)
img | Rotate{}.angle(45.0f).scale(1.0f)
img | Pad{}.top(10).bottom(10).left(10).right(10).mode(PadMode::Constant)
img | PadToSquare{}
img | WarpAffine{}.matrix(M).width(w).height(h)
img | WarpPerspective{}.homography(H).width(w).height(h)
```

### Filter & Enhancement

```cpp
img | GaussianBlur{}.kernel_size(3)
img | MedianBlur{}.kernel_size(5)
img | BilateralFilter{}.diameter(9).sigma_color(75).sigma_space(75)
img | UnsharpMask{}.sigma(1.0f).strength(1.5f)
img | CLAHE{}.clip_limit(2.0).tile_size(8)
img | GammaCorrection{}.gamma(0.5f)            // < 1 brightens, > 1 darkens
```

### Morphology & Threshold

```cpp
img | Dilate{}.kernel_size(3).iterations(1)
img | Erode{}.kernel_size(3)
img | Threshold{}.value(128.0).mode(ThresholdMode::Binary)
img | Threshold{}.mode(ThresholdMode::Otsu)    // automatic threshold
```

### Edge Detection

```cpp
// Both accept Image<Gray> or Image<BGR> (auto-converted internally)
img | SobelEdge{}.ksize(3)                     // returns Image<Gray>
img | CannyEdge{}.threshold1(50).threshold2(150)
```

### Normalization

```cpp
img | Normalize{}                              // [0, 255] → [0, 1]
img | NormalizeTo{0.0f, 1.0f}                  // explicit range
img | Standardize{}.mean(0.485f).std(0.229f)   // per-channel mean/std
```

### Color

```cpp
img | Brightness{}.delta(20.0)
img | Contrast{}.factor(1.2f)
img | WeightedBlend{other}.alpha(0.7f)
img | AlphaBlend{overlay}                      // overlay is Image<BGRA>
img | ApplyMask{mask}                          // mask is Image<Gray>
```

## Lazy Views over Collections

```cpp
#include "improc/views/views.hpp"
namespace views = improc::views;

std::vector<Image<BGR>> images = ...;

// transform every image lazily
auto resized = images
    | views::transform(Resize{}.width(64).height(64))
    | views::to<std::vector<Image<BGR>>>();

// filter, then transform, then take first 10
auto result = images
    | views::filter([](const Image<BGR>& img) { return img.cols() >= 128; })
    | views::transform(Resize{}.width(64).height(64))
    | views::take(10)
    | views::to<std::vector<Image<BGR>>>();

// iterate lazily over a directory — images loaded on demand
for (const auto& img : views::from_dir("dataset/", {".png", ".jpg"})) {
    // process img — only this image is in RAM at a time
}

// batch(N) — process in fixed-size chunks
for (const auto& chunk : views::from_dir("frames/", {".png"}) | views::batch(8)) {
    // chunk is std::vector<Image<BGR>>, up to 8 images
}

// enumerate — zero-based index alongside each element
for (const auto& [idx, img] : images | views::enumerate) {
    std::cout << idx << ": " << img.cols() << "x" << img.rows() << "\n";
}

// zip — pair two sources element-wise
for (const auto& [img, mask] : views::zip(images, masks)) {
    Image<BGR> masked = img | ApplyMask{mask};
}
```

## Augmentation

Stochastic augmentations for training pipelines. Each op takes `(Image<F>, std::mt19937&)` directly or via `.bind(rng)` for `operator|` use.

```cpp
#include "improc/ml/augmentation.hpp"
using namespace improc::ml;

std::mt19937 rng(42);

// Single op — direct call
Image<BGR> flipped = RandomFlip{}.p(0.5f)(img, rng);

// Pipeline form via .bind()
Image<BGR> augmented = img
    | RandomFlip{}.p(0.5f).bind(rng)
    | RandomBrightness{}.range(0.8f, 1.2f).bind(rng)
    | RandomRotate{}.range(-15.0f, 15.0f).bind(rng);

// Full training augmentor with composition ops
auto augmentor = Compose<BGR>{}
    .add(RandomFlip{}.p(0.5f))
    .add(RandomApply<BGR>{ColorJitter{}, 0.5f})
    .add(OneOf<BGR>{}
        .add(RandomGaussianNoise{}.std_dev(5.0f, 15.0f))
        .add(RandomSaltAndPepper{}.p(0.02f)));

// Apply once
Image<BGR> out = augmentor(img, rng);

// Or in a pipeline
Image<BGR> out2 = img | augmentor.bind(rng);
```

All augmentation ops are constrained by C++20 concepts — passing the wrong format is a compiler error.

## ML Preprocessing Pipeline

A complete image-to-tensor pipeline for a classification model:

```cpp
using namespace improc::core;
using namespace improc::io;

auto src = imread<BGR>("photo.jpg");
if (!src) return 1;

// Standard ImageNet preprocessing
Image<Float32C3> tensor = *src
    | Resize{}.width(224).height(224)
    | CLAHE{}.clip_limit(2.0)
    | ToFloat32C3{}
    | NormalizeTo{0.0f, 1.0f};
```

## Next Steps

- [Real-Time Camera](real-time-camera.md) — `CameraCapture` + `ThreadPool` + `FramePipeline`
- [NAMESPACES.md](../../NAMESPACES.md) — complete API reference for every op, error code, and return type
