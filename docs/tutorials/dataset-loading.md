# Dataset Loading

This tutorial shows how to load image datasets for training: the simple `Dataset` and `ImageLoader` for folder-based classification datasets, and `VocDataset` / `CocoDataset` for detection datasets in standard annotation formats.

## Prerequisites

- Completed [Getting Started](getting-started.md)
- `improc/ml/ml.hpp` (umbrella include)

## ImageLoader — Load a Flat Directory

`ImageLoader` loads every `.jpg`, `.jpeg`, and `.png` from a directory into `cv::Mat` objects. Good for unsorted collections or pre-split datasets.

```cpp
#include "improc/ml/ml.hpp"
using namespace improc::ml;

ImageLoader loader{};
loader.load_images("data/unlabeled/");

auto result = loader.get_images();
if (!result) {
    std::cerr << result.error().message << "\n";
    return 1;
}

std::vector<cv::Mat>& images = *result;
std::cout << "Loaded " << images.size() << " images\n";
```

`load_images` throws `FileNotFoundError` if the path is not a directory. `get_images()` returns an error if no images were found.

## Dataset — Class-Labelled Folder Structure

`Dataset` expects one subdirectory per class:

```
data/animals/
  cats/   image1.png  image2.png  ...
  dogs/   image3.png  ...
  birds/  image4.png  ...
```

It loads all images, shuffles them, and splits into train/val/test sets.

```cpp
#include "improc/ml/ml.hpp"
using namespace improc::ml;

Dataset ds{};
ds.set_shuffle_seed(42);

auto ok = ds.load_from_directory(
    "data/animals/",
    /*test_ratio=*/  0.2f,   // 20 % test
    /*val_ratio=*/   0.1f,   // 10 % validation
    /*max_per_class=*/std::nullopt  // no per-class cap
);

if (!ok) { std::cerr << ok.error().message << "\n"; return 1; }

std::cout << "Train: " << ds.train_images().size() << "\n";
std::cout << "Val:   " << ds.val_images().size()   << "\n";
std::cout << "Test:  " << ds.test_images().size()  << "\n";

// Integer label → class name
for (std::size_t i = 0; i < ds.train_images().size(); ++i) {
    int  lbl  = ds.train_labels()[i];
    auto name = ds.class_name_for(lbl);
    std::cout << name << "\n";
}

// Full class mapping
for (const auto& [name, id] : ds.class_mapping())
    std::cout << id << " → " << name << "\n";
```

Cap the number of images per class to keep training fast:

```cpp
ds.load_from_directory("data/animals/", 0.2f, 0.1f, /*max_per_class=*/500);
```

## VocDataset — Pascal VOC Detection Format

Expected directory layout (standard Pascal VOC 2007/2012):

```
VOC2012/
  Annotations/           (*.xml per image)
  JPEGImages/            (*.jpg images)
  ImageSets/Main/        (train.txt, val.txt, test.txt — optional)
```

If `ImageSets/Main/` is present, the official split is used. Otherwise the dataset is split randomly by ratio.

```cpp
#include "improc/ml/ml.hpp"
using namespace improc::ml;

VocDataset ds{};
ds.classes({"cat", "dog", "bird"})   // optional: fix order + filter unknowns
  .skip_difficult(true)              // skip objects with <difficult>1</difficult>
  .shuffle_seed(42);                 // only used in random-split mode

auto ok = ds.load_from_directory("data/VOC2012/");
if (!ok) { std::cerr << ok.error().message << "\n"; return 1; }

std::cout << "Train: " << ds.train().size() << "\n";
std::cout << "Val:   " << ds.val().size()   << "\n";

// AnnotatedImage<BGR> = {Image<BGR> image, std::vector<BBox> boxes}
for (const auto& sample : ds.train()) {
    for (const auto& box : sample.boxes) {
        std::cout << box.label << " @ ("
                  << box.box.x << ", " << box.box.y << ")\n";
    }
}

// Class id → name
std::cout << ds.class_name_for(0) << "\n";
```

Omit `.classes()` to auto-assign IDs from all classes found in the XML files.

## CocoDataset — COCO JSON Format

Expected directory layout:

```
coco/
  annotations/
    instances_train2017.json
    instances_val2017.json
  images/
    train2017/
    val2017/
```

Each split is loaded independently with an explicit call. All splits share one consistent class mapping — call `.classes()` **before** any `load_*` call to fix the class order.

```cpp
#include "improc/ml/ml.hpp"
using namespace improc::ml;

CocoDataset ds{};
ds.classes({"person", "car", "bicycle"})  // MUST be before load_* if filtering
  .skip_crowd(true);                      // skip iscrowd=1 annotations

auto ok_train = ds.load_train(
    "coco/annotations/instances_train2017.json",
    "coco/images/train2017/");
if (!ok_train) { std::cerr << ok_train.error().message << "\n"; return 1; }

auto ok_val = ds.load_val(
    "coco/annotations/instances_val2017.json",
    "coco/images/val2017/");

std::cout << "Train: " << ds.train().size() << "\n";
std::cout << "Val:   " << ds.val().size()   << "\n";

for (const auto& sample : ds.val()) {
    for (const auto& box : sample.boxes)
        std::cout << box.label << "  id=" << box.class_id << "\n";
}
```

Non-contiguous COCO category IDs (e.g. 1, 2, 3, 44, …) are remapped to sequential 0-indexed IDs consistent across all three splits.

## Low-Level Parsing Functions

For custom loading logic, use the free-function parsers directly:

```cpp
// Parse one VOC XML file
std::unordered_map<std::string, int> class_map;
auto ann = parse_voc_xml(
    "VOC2012/Annotations/2007_000032.xml",
    "VOC2012/JPEGImages/",
    class_map,
    /*skip_difficult=*/ true,
    /*filter_unknown=*/ false);
if (ann) {
    for (const auto& b : ann->boxes)
        std::cout << b.label << "\n";
}

// Parse one COCO JSON file
auto samples = parse_coco_json(
    "coco/annotations/instances_val2017.json",
    "coco/images/val2017/",
    class_map);
if (samples)
    std::cout << "Loaded " << samples->size() << " images\n";
```

## Using Datasets with Views

`views::from_dir` and the dataset classes work naturally together for lazy preprocessing:

```cpp
#include "improc/views/views.hpp"
namespace views = improc::views;

// Lazy resize of all training images
auto preprocessed = ds.train()
    | views::transform([](const AnnotatedImage<BGR>& s) {
        return AnnotatedImage<BGR>{
            s.image | Resize{}.width(640).height(640),
            s.boxes
        };
      })
    | views::to<std::vector<AnnotatedImage<BGR>>>();
```

## Next Steps

- [Augmentation Pipeline](augmentation-pipeline.md) — `BBoxCompose` and `VocSegDataset`
- [ML Inference](ml-inference.md) — running a model on your loaded images
- [Evaluation Metrics](evaluation-metrics.md) — measuring accuracy on your test set
