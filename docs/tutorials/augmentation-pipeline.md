# Augmentation Pipeline

This tutorial shows how to build augmentation pipelines for training data using improc++. Topics covered: geometric and colour ops, composition helpers (`Compose`, `RandomApply`, `OneOf`), and mix-based augmentation (`MixUp`, `CutMix`, `MixCompose`).

## Prerequisites

- Completed [Getting Started](getting-started.md)
- Basic familiarity with `Image<Format>` and `std::mt19937`

## Basic Ops

All augmentation ops live in `improc::ml` (include `improc/ml/augmentation.hpp`). Each op takes `(Image<Format> img, std::mt19937& rng)` and returns the transformed image. Randomness is always explicit — you own the RNG.

```cpp
#include "improc/io/image_io.hpp"
#include "improc/ml/augmentation.hpp"

using namespace improc::ml;
using namespace improc::core;

int main() {
    std::mt19937 rng{42};

    auto src = improc::io::imread<BGR>("photo.png");
    if (!src) { std::cerr << src.error().message << "\n"; return 1; }

    // Geometric
    HorizontalFlip flip{};
    auto flipped = flip(*src, rng);

    RandomRotate rot{};
    rot.max_angle(30.0f);          // ±30°
    auto rotated = rot(*src, rng);

    RandomCrop crop{};
    crop.min_scale(0.7f).max_scale(1.0f);
    auto cropped = crop(*src, rng);

    // Colour
    ColorJitter jitter{};
    jitter.brightness(0.3f).contrast(0.3f).saturation(0.2f);
    auto jittered = jitter(*src, rng);

    // Noise
    GaussianNoise noise{};
    noise.stddev(15.0f);
    auto noisy = noise(*src, rng);

    // Blur
    GaussianBlur blur{};
    blur.max_ksize(7);
    auto blurred = blur(*src, rng);
}
```

## Composing Ops with `Compose`

`Compose<Format>` chains ops sequentially. Each step receives the output of the previous one.

```cpp
#include "improc/ml/augmentation.hpp"
using namespace improc::ml;

std::mt19937 rng{42};

Compose<BGR> pipeline{};
pipeline
    .add(HorizontalFlip{})
    .add(RandomRotate{}.max_angle(15.0f))
    .add(ColorJitter{}.brightness(0.2f).contrast(0.2f))
    .add(GaussianNoise{}.stddev(10.0f));

auto aug = pipeline(*src, rng);

// Equivalent pipeline form:
auto aug2 = *src | pipeline.bind(rng);
```

## Applying an Op with Some Probability

`RandomApply<Format>` wraps a single op and applies it only `p` fraction of the time.

```cpp
// Apply horizontal flip 50 % of the time
RandomApply<BGR> maybe_flip{HorizontalFlip{}, 0.5f};

auto aug = maybe_flip(*src, rng);
```

`RandomApply` composes naturally inside `Compose`:

```cpp
Compose<BGR> pipeline{};
pipeline
    .add(RandomApply<BGR>{HorizontalFlip{}, 0.5f})
    .add(RandomApply<BGR>{RandomRotate{}.max_angle(20.0f), 0.3f})
    .add(ColorJitter{}.brightness(0.4f));
```

## Choosing One Op at Random with `OneOf`

`OneOf<Format>` picks a single augmentation uniformly at random from its list.

```cpp
OneOf<BGR> color_aug{};
color_aug
    .add(ColorJitter{}.saturation(0.5f))
    .add(GaussianBlur{}.max_ksize(5))
    .add(RandomErasing<BGR>{});  // randomly erases a patch

auto aug = color_aug(*src, rng);
```

## Full Training Pipeline Example

```cpp
#include "improc/ml/augmentation.hpp"
#include "improc/ml/ml.hpp"
#include "improc/io/image_io.hpp"
using namespace improc::ml;
using namespace improc::io;

int main() {
    std::mt19937 rng{std::random_device{}()};

    // Build a reusable pipeline
    Compose<BGR> train_aug{};
    train_aug
        .add(RandomApply<BGR>{HorizontalFlip{}, 0.5f})
        .add(RandomApply<BGR>{RandomRotate{}.max_angle(15.0f), 0.3f})
        .add(RandomApply<BGR>{RandomCrop{}.min_scale(0.8f).max_scale(1.0f), 0.5f})
        .add(ColorJitter{}.brightness(0.3f).contrast(0.3f).saturation(0.15f))
        .add(RandomApply<BGR>{GaussianNoise{}.stddev(8.0f), 0.2f});

    // Apply to a batch
    Dataset ds{"data/train"};
    for (auto& entry : ds) {
        auto img = imread<BGR>(entry.path);
        if (!img) continue;
        auto augmented = train_aug(*img, rng);
        // ... feed to model
    }
}
```

## MixUp

`MixUp` blends two images and their one-hot labels with a Beta(α, α)-distributed weight. Inputs must be the same size and have equal-length label vectors.

```cpp
#include "improc/ml/augmentation.hpp"
using namespace improc::ml;

std::mt19937 rng{42};

MixUp mixup{};
mixup.alpha(0.4f)   // Beta distribution parameter (default: 0.4)
     .p(0.5f);      // Probability of applying (default: 1.0)

// LabeledImage<F> = {Image<F> image, std::vector<float> label}
LabeledImage<BGR> a{ *imread<BGR>("cat.png"),  {1.f, 0.f, 0.f} };
LabeledImage<BGR> b{ *imread<BGR>("dog.png"),  {0.f, 1.f, 0.f} };

auto mixed = mixup(a, b, rng);
// mixed.label ≈ {0.55, 0.45, 0.0}  (depends on sampled λ)
```

## CutMix

`CutMix` pastes a rectangular region from a second image into the first, then adjusts labels proportionally to the pasted area.

```cpp
CutMix cutmix{};
cutmix.alpha(1.0f)   // Beta parameter for patch-size ratio
      .p(0.5f);

auto mixed = cutmix(a, b, rng);
// Box region from b is inserted; label reflects actual mixed area ratio
```

## MixCompose — Chaining Mix Ops

`MixCompose<Format>` applies a sequence of mix operations (each takes primary + secondary + rng) in order.

```cpp
MixCompose<BGR> mix_pipeline{};
mix_pipeline
    .add(MixUp{}.alpha(0.4f).p(0.5f))
    .add(CutMix{}.alpha(1.0f).p(0.5f));

auto result = mix_pipeline(a, b, rng);

// Pipeline form — b and rng must outlive the pipeline binding:
auto result2 = a | mix_pipeline.bind(b, rng);
```

## Bounding-Box-Aware Augmentation

For object detection datasets, use `BBoxCompose` which updates bounding boxes alongside the image.

```cpp
#include "improc/ml/augmentation.hpp"
using namespace improc::ml;

BBoxCompose<BGR> det_aug{};
det_aug
    .add(RandomHorizontalFlip<BGR>{0.5f})
    .add(RandomScale<BGR>{}.min_scale(0.8f).max_scale(1.2f))
    .add(ColorJitter{}.brightness(0.3f));

// AnnotatedImage<F> = {Image<F> image, std::vector<BBox> boxes}
AnnotatedImage<BGR> sample{ *imread<BGR>("frame.png"), boxes };
auto aug_sample = det_aug(sample, rng);
// aug_sample.boxes are transformed/clipped to match the new image
```

## Error Handling

Invalid parameters throw `improc::ParameterError`:

```cpp
try {
    RandomApply<BGR> bad{HorizontalFlip{}, 1.5f};  // p > 1.0
} catch (const improc::ParameterError& e) {
    std::cerr << e.what() << "\n";
}

try {
    MixUp{}.alpha(-1.0f);  // alpha must be > 0
} catch (const improc::ParameterError& e) {
    std::cerr << e.what() << "\n";
}
```

## Next Steps

- [ML Inference](ml-inference.md) — run the trained model on new images
- [Evaluation Metrics](evaluation-metrics.md) — measure accuracy, mAP, and tracking quality
- [ML Visualization Charts](ml-visualization-charts.md) — plot confusion matrices and PR curves
