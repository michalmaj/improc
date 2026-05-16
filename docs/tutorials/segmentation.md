# Segmentation

This tutorial covers pixel-level image segmentation in improc++: the `SegmentedImage<F>` type, segmentation-aware augmentation with `SegCompose`, loading Pascal VOC segmentation datasets with `VocSegDataset`, and measuring quality with `SegEval`.

## Prerequisites

- Completed [Augmentation Pipeline](augmentation-pipeline.md)
- Completed [Dataset Loading](dataset-loading.md)
- `improc/ml/augmentation.hpp`, `improc/ml/ml.hpp`

## SegmentedImage

`SegmentedImage<F>` pairs an image with its segmentation masks:

```cpp
template<AnyFormat Format>
struct SegmentedImage {
    Image<Format>              image;        // the RGB/BGR image
    Image<Gray>                class_mask;   // pixel = class_id; 255 = void/unlabelled
    std::optional<Image<Gray>> instance_mask; // pixel = instance_id; nullopt if not loaded
};
```

The pixel value in `class_mask` is the integer class ID. Pixels with value `255` are void (ignore during training and evaluation).

### Build a SegmentedImage manually

```cpp
#include "improc/ml/ml.hpp"
using namespace improc::ml;
using namespace improc::core;

auto img  = *imread<BGR>("frame.png");
auto mask = *imread<Gray>("mask.png");   // pixel = class_id, 255 = void

SegmentedImage<BGR> sample{ img, mask };

// With instance mask
auto inst = *imread<Gray>("instance_mask.png");
SegmentedImage<BGR> full{ img, mask, inst };
```

## VocSegDataset — Pascal VOC Segmentation

Expected layout:

```
VOCdevkit/VOC2012/
  JPEGImages/               (*.jpg images)
  SegmentationClass/        (*.png class masks, required)
  SegmentationObject/       (*.png instance masks, optional)
  ImageSets/Segmentation/   (train.txt, val.txt — optional)
```

```cpp
#include "improc/ml/ml.hpp"
using namespace improc::ml;

VocSegDataset ds{};
ds.classes({              // index = class_id
    "background",         // 0
    "aeroplane",          // 1
    "bicycle",            // 2
    "bird",               // 3
    "cat",                // 8
    "dog",                // 12
    // ... up to 20 VOC classes
  })
  .load_instance_masks(true);   // also load SegmentationObject/ (default: false)

auto ok = ds.load_from_directory("VOCdevkit/VOC2012/");
if (!ok) { std::cerr << ok.error().message << "\n"; return 1; }

std::cout << "Train: " << ds.train().size() << "\n";
std::cout << "Val:   " << ds.val().size()   << "\n";

// Iterate
for (const auto& sample : ds.train()) {
    // sample.image        — Image<BGR>
    // sample.class_mask   — Image<Gray>, pixel = class_id, 255 = void
    // sample.instance_mask — std::optional<Image<Gray>>
    std::cout << "Image: " << sample.image.cols() << "x" << sample.image.rows() << "\n";
}

// Class name lookup
std::cout << ds.class_name_for(8) << "\n";  // "cat"
```

Omit `.classes()` to load without name mapping — `class_name_for()` will throw for any ID.

## Segmentation-Aware Augmentation

Standard augmentation ops that modify geometry (flip, rotate, crop, etc.) have `SegmentedImage<F>` overloads. The masks are transformed alongside the image using `cv::INTER_NEAREST` to preserve integer class IDs. Colour augmentations (brightness, jitter, etc.) pass the masks through unchanged.

```cpp
#include "improc/ml/augmentation.hpp"
using namespace improc::ml;

std::mt19937 rng{42};

SegmentedImage<BGR> sample = /* from VocSegDataset or manually constructed */;

// Single op — direct call
SegmentedImage<BGR> flipped = RandomFlip<BGR>{0.5f}(sample, rng);

// Full pipeline with SegCompose
SegCompose<BGR> aug{};
aug.add(RandomFlip<BGR>{0.5f})
   .add(RandomRotate<BGR>{}.max_angle(15.0f))
   .add(RandomCrop<BGR>{}.min_scale(0.8f).max_scale(1.0f))
   .add(ColorJitter{}.brightness(0.3f).contrast(0.3f));

SegmentedImage<BGR> augmented = aug(sample, rng);

// Pipeline form
SegmentedImage<BGR> aug2 = sample | aug.bind(rng);
```

`SegCompose<F>` works exactly like `BBoxCompose<F>` or `Compose<F>` — chain ops with `.add()` and execute with `operator()` or `.bind(rng)` for pipeline use.

## Training Loop Example

```cpp
#include "improc/ml/ml.hpp"
#include "improc/ml/augmentation.hpp"
using namespace improc::ml;

VocSegDataset ds{};
ds.classes({"background", "person", "car" /*, ... */});
ds.load_from_directory("VOCdevkit/VOC2012/");

std::mt19937 rng{std::random_device{}()};

SegCompose<BGR> train_aug{};
train_aug
    .add(RandomFlip<BGR>{0.5f})
    .add(RandomRotate<BGR>{}.max_angle(10.0f))
    .add(RandomCrop<BGR>{}.min_scale(0.7f).max_scale(1.0f))
    .add(ColorJitter{}.brightness(0.2f).saturation(0.1f));

for (const auto& sample : ds.train()) {
    auto aug_sample = train_aug(sample, rng);

    // aug_sample.image      — augmented BGR image
    // aug_sample.class_mask — correspondingly transformed class mask
    // Feed both into your segmentation model...
}
```

## Evaluating Segmentation Quality

`SegEval` accumulates per-frame predictions and computes mIoU, mean Dice, and per-class IoU/Dice.

```cpp
#include "improc/ml/eval/eval.hpp"
using namespace improc::ml;

SegEval eval{};
eval.num_classes(21);   // VOC has 21 classes (incl. background)

for (const auto& sample : ds.val()) {
    // Run your segmentation model to get a predicted mask
    Image<Gray> pred_mask = run_segmentation_model(sample.image);

    eval.update(pred_mask, sample.class_mask);
    // void pixels (255) in sample.class_mask are automatically ignored
}

auto metrics = eval.compute();
std::cout << "mIoU:       " << metrics.mIoU      << "\n";
std::cout << "Mean Dice:  " << metrics.mean_dice  << "\n";

for (const auto& [class_id, iou] : metrics.per_class_iou)
    std::cout << ds.class_name_for(class_id) << "  IoU=" << iou << "\n";
```

### One-off pixel IoU and Dice

```cpp
float iou  = pixel_iou(pred_mask, gt_mask, /*class_id=*/8);   // "cat"
float dice = dice(pred_mask, gt_mask, 8);
```

Both functions ignore pixels where `gt_mask == 255` (void).

## Low-Level Parser

```cpp
// Parse one sample manually
auto sample = parse_voc_seg(
    "2007_000032",                          // stem (no extension)
    "VOC2012/JPEGImages/",
    "VOC2012/SegmentationClass/",
    "VOC2012/SegmentationObject/");  // empty path → no instance mask

if (sample)
    std::cout << "Class mask size: "
              << sample->class_mask.cols() << "x" << sample->class_mask.rows() << "\n";
```

## Visualizing a Class Mask

Class masks are single-channel with integer pixel values. To inspect them:

```cpp
// Colorize: map each class_id to a distinct BGR colour
cv::Mat mask = sample.class_mask.mat();
cv::Mat colored(mask.size(), CV_8UC3, cv::Scalar{0, 0, 0});

std::vector<cv::Scalar> palette = {
    {0, 0, 0},       // background
    {128, 0, 0},     // aeroplane
    {0, 128, 0},     // bicycle
    // ... one entry per class
};

for (int r = 0; r < mask.rows; ++r)
    for (int c = 0; c < mask.cols; ++c) {
        uint8_t cls = mask.at<uint8_t>(r, c);
        if (cls < palette.size())
            colored.at<cv::Vec3b>(r, c) = cv::Vec3b(
                palette[cls][0], palette[cls][1], palette[cls][2]);
    }

Image<BGR>{colored} | Show{"Class Mask"};
```

## Next Steps

- [Evaluation Metrics](evaluation-metrics.md) — `SegEval`, `DetectionEval`, `ClassEval`
- [Augmentation Pipeline](augmentation-pipeline.md) — `BBoxCompose`, `MixUp`, `CutMix`
- [Dataset Loading](dataset-loading.md) — `VocDataset`, `CocoDataset` for detection
