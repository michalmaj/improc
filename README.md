# improc++: Modern C++ Image Processing Toolkit

[![CI](https://github.com/michalmaj/improc/actions/workflows/ci.yml/badge.svg)](https://github.com/michalmaj/improc/actions/workflows/ci.yml)

> **Who is this for?**  
> C++ engineers and researchers building real-time computer vision and ML systems who want
> compile-time safety, composable pipelines, and ML-ready utilities — without leaving the OpenCV ecosystem.

## Table of Contents

- [Status](#status)
- [Getting Started](#getting-started)
- [API Comparison](#api-comparison)
- [Features](#features)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Augmentation](#augmentation)
- [Video Recording](#video-recording)
- [ONNX Runtime Inference](#onnx-runtime-inference)
- [Edge Detection & Enhancement](#edge-detection--enhancement)
- [Drawing Detections](#drawing-detections)
- [Object Tracking](#object-tracking)
- [Lazy Views](#lazy-views)
- [Feature Detection Pipeline](#feature-detection-pipeline)
- [Requirements](#requirements)
- [Installation](#installation)
- [CMake Configuration (CLion)](#cmake-configuration-clion)
- [Running Tests](#running-tests)
- [Tested With](#tested-with)
- [Non-Goals & Limitations](#non-goals--limitations)
- [Motivation](#motivation)
- [Contributing](#contributing)

## Status

> **Latest release: v0.5.0** — Multi-Object Tracking. `improc::ml` now ships `IouTracker`, `SortTracker` (Kalman + Hungarian, SORT algorithm), and `ByteTracker` (two-stage BYTE algorithm) plus `TrackingEval` for MOTA / MOTP / IDF1 evaluation.  
> **Previous highlights:** v0.4.0 — extended augmentation, bbox-aware ops, VOC/COCO dataset loaders, segmentation support, lazy views, ONNX Runtime 1.20.1. v0.3.0 — complete classical 2D CV pipeline.  
> APIs are stabilising but may still change between minor versions.

| Namespace | Status | Notes |
|---|---|---|
| `improc::core` | ✅ Stable | New ops added regularly |
| `improc::io` | ✅ Stable | |
| `improc::ml` | ✅ Stable | New ops added regularly |
| `improc::threading` | ✅ Stable | |
| `improc::visualization` | ✅ Stable | New ops added regularly |
| `improc::views` | ✅ Stable | Lazy image pipeline adapters |
| `improc::onnx` | ✅ Stable | ONNX Runtime 1.20.1; CPU + CoreML on Apple Silicon |
| `improc::cuda` | 🔜 Planned | GPU-accelerated ops via OpenCV CUDA |

## Getting Started

### Prerequisites

- C++23 compiler: GCC 14+ or Clang 18+
- CMake 3.30+
- [Conan 2](https://conan.io/): `pip install "conan>=2,<3"`
- macOS: Xcode Command Line Tools (`xcode-select --install`)
- Linux: `sudo apt-get install build-essential cmake libgtk-3-dev`

### Build

```bash
# 1. Detect your compiler profile (run once)
conan profile detect --force

# 2. Configure — Conan installs OpenCV, GTest, Eigen automatically
cmake \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" \
  -DCONAN_COMMAND=$(which conan) \
  -DCMAKE_BUILD_TYPE=Release \
  -B build .

# 3. Build
cmake --build build --parallel

# 4. Run tests
./build/improc_tests
```

### Your First Pipeline

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"
using namespace improc::core;
using namespace improc::io;

int main() {
    auto img = imread<BGR>("photo.png");
    if (!img) { return 1; }

    Image<BGR> result = *img
        | Resize{}.width(224).height(224)
        | GaussianBlur{}.kernel_size(3)
        | Brightness{}.delta(20.0);

    imwrite("output.png", result);
}
```

## API Comparison

| Task | Raw OpenCV | improc++ |
|---|---|---|
| Resize to 224×224 | `cv::resize(src, dst, cv::Size(224,224))` | `img \| Resize{}.width(224).height(224)` |
| Convert to float | `src.convertTo(dst, CV_32FC3, 1.0/255)` | `img \| ToFloat32C3{} \| NormalizeTo{0.f, 1.f}` |
| Gaussian blur | `cv::GaussianBlur(src, dst, cv::Size(3,3), 0)` | `img \| GaussianBlur{}.kernel_size(3)` |
| Edge detection | `cv::Canny(gray, dst, 100, 200)` | `img \| CannyEdge{}.threshold1(100).threshold2(200)` |
| Warp perspective | `cv::warpPerspective(src, dst, H, cv::Size(w,h))` | `img \| WarpPerspective{}.homography(H).width(w).height(h)` |

Format mismatches (e.g. passing `Image<Float32>` where `Image<BGR>` is expected) are **compiler errors**, not silent runtime bugs.

## Non-Goals & Limitations

- **Not a Python library yet.** Python bindings are on the long-term roadmap — use OpenCV's Python API in the meantime.
- **Not a general-purpose image editor.** No GUI, no layers, no undo — this is a processing pipeline toolkit.
- **Not a training framework.** No autograd, no loss functions — use PyTorch/TensorFlow for training; improc++ handles inference and preprocessing.
- **No CUDA yet.** All ops run on CPU (ONNX Runtime uses CoreML EP on Apple Silicon). CUDA acceleration (`improc::cuda`) is planned.
- **OpenCV dependency is visible.** `cv::Mat`, `cv::Point2f` etc. appear in the public API — improc++ wraps, not hides, OpenCV.
- **C++23 required.** No support for older standards.

## Motivation

OpenCV is powerful but its raw API is stringly-typed, mutation-heavy, and easy to misuse — passing a `CV_32FC1` mat where a `CV_8UC3` is expected silently produces garbage instead of a compiler error. improc++ wraps OpenCV with a thin, low-overhead abstraction that makes format mismatches impossible at compile time, composes processing steps into readable pipelines, and provides ML-ready utilities (augmentation, dataset loading, DNN inference) without reinventing the wheel. Benchmarks on Apple M4 Pro show wrapper overhead of ≤0.9% for most ops; CLAHE and GammaCorrection carry additional per-call cost from object/LUT construction in the current implementation. The goal is code that reads like a description of what it does, not how OpenCV internals work.

## Features

- **Type-safe image wrapper** — `Image<BGR>`, `Image<Gray>`, `Image<Float32>` etc. catch format mismatches at compile time
- **Composable pipeline** — `image | Resize{}.width(224) | ToFloat32C3{}` syntax for readable processing chains
- **Geometric operations** — `Resize` (aspect-ratio aware), `Crop`, `CenterCrop`, `LetterBox`, `Flip`, `Rotate`, `Pad`, `PadToSquare`, `WarpAffine`, `WarpPerspective`, `find_homography`
- **Filter & morphology** — `GaussianBlur`, `MedianBlur`, `Dilate`, `Erode`, `MorphOpen`, `MorphClose`, `MorphGradient`, `TopHat`, `BlackHat`; `Threshold` (incl. Otsu & adaptive), `CLAHE`, `GammaCorrection`, `BilateralFilter`, `HistogramEqualization`, `NLMeansDenoising`, `UnsharpMask`
- **Edge & corner detection** — `SobelEdge`, `CannyEdge`, `LaplacianEdge`, `HarrisCorner`; all accept `Image<Gray>` or `Image<BGR>` (auto-converted)
- **Colour spaces** — `ToHSV`, `ToLAB`, `ToYCrCb`, `ToGray`, `ToBGR` (from any of the above); format tags `HSV`, `LAB`, `YCrCb` enforce colour-space safety at compile time
- **Pyramid ops** — `PyrDown` (halves), `PyrUp` (doubles); work on any `Image<Format>`
- **Drawing / annotation** — `DrawText`, `DrawLine`, `DrawCircle`, `DrawRectangle`; all operate on a clone (non-mutating)
- **Contour analysis** — `FindContours` → `ContourSet` (with `area()`, `perimeter()`, `bounding_rect()`); `DrawContours` (fill or stroke, single or all)
- **Connected components** — `ConnectedComponents` → `ComponentMap` (labels, stats, centroids, per-component masks); `DistanceTransform` → `Image<Float32>`
- **Feature detection pipeline** — `DetectORB` / `DetectSIFT` / `DetectAKAZE` → `KeypointSet`; `DescribeORB` / `DescribeSIFT` / `DescribeAKAZE` → `DescriptorSet`; `MatchBF` / `MatchFlann` → `MatchSet`; `DrawKeypoints` / `DrawMatches` for visualisation
- **Pixel ops** — `InRange` (binary mask from channel bounds), `Invert`, `Brightness`, `Contrast`, `WeightedBlend`, `AlphaBlend`, `ApplyMask`
- **Normalization** — `Normalize`, `NormalizeTo`, `Standardize` for ML preprocessing
- **Format conversions** — explicit, compiler-enforced free functions (`convert<Gray>(bgr_image)`)
- **Augmentation** — stochastic training augmentations constrained by C++20 concepts: `RandomFlip`, `RandomRotate`, `RandomCrop`, `RandomResize`, `RandomZoom`, `RandomShear`, `RandomPerspective`, `RandomBrightness`, `RandomContrast`, `ColorJitter`, `RandomGrayscale`, `RandomSolarize`, `RandomPosterize`, `RandomEqualize`, `RandomBlur`, `RandomSharpness`, `RandomGaussianNoise`, `RandomSaltAndPepper`, `RandomErasing`, `GridDropout`, `Compose`, `RandomApply`, `OneOf`; bbox-aware overloads via `BBox`, `AnnotatedImage<F>`, `BBoxCompose<F>` (all 7 geometric ops accept annotated images with automatic clip-and-drop filtering)
- **Multi-object tracking** — `IouTracker` (greedy IoU), `SortTracker` (constant-velocity Kalman + Hungarian, SORT algorithm), `ByteTracker` (two-stage BYTE algorithm: Stage 1 Hungarian on high-confidence dets, Stage 2 greedy IoU on low-confidence dets); all satisfy `TrackerType<T>` and are drop-in replaceable; `TrackingEval` accumulates MOTA, MOTP, IDF1, Precision, Recall, FP, FN, IDSW across frames
- **Dataset loading** — load image datasets from class-labeled directories with train/val/test splitting
- **DNN inference** — `DnnClassifier`, `DnnDetector` (YOLO & SSD), `DnnForward` backed by OpenCV DNN; pipeline-composable
- **ONNX Runtime inference** — `OnnxClassifier`, `OnnxDetector` (YOLOv5 & YOLOv8 & SSD), `OnnxSession` (raw tensor I/O); CPU + CoreML EP on Apple Silicon; train in Python, export to ONNX, run here
- **Camera capture** — asynchronous threaded frame capture via `CameraCapture`; `getFrame()` returns `std::expected<cv::Mat, Error>` for safe error handling
- **Video recording** — synchronous RAII `VideoWriter` with auto codec detection and pipeline support (`img | Show{"preview"} | writer`)
- **Haar Cascade loader** — CRTP-based model loader for OpenCV cascade classifiers
- **Threading** — `ThreadPool` and `FramePipeline<Result>` for real-time frame processing
- **Visualization** — `Histogram`, `LinePlot`, `Scatter` chart functors, a `Show` passthrough display op, `DrawBoundingBoxes` for annotating detections, `DrawTracks` for annotating tracker output, and `Montage` for image grid composition — all composable with `operator|`
- **Lazy image views** — `improc::views` lazy pipeline adapters: `transform`, `filter`, `take`, `drop`, `batch(N)`, `enumerate`, `zip`; compose with `from_dir()`, `VideoView`, and `std::vector<Image<F>>` sources via `operator|`; zero work until materialised with `views::to<T>()`

## Quick Start

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/camera_capture.hpp"
#include "improc/io/video_writer.hpp"
#include "improc/visualization/visualization.hpp"

using namespace improc::core;
using namespace improc::io;
using namespace improc::visualization;

// 1. Load an image and run a preprocessing pipeline
cv::Mat raw = cv::imread("photo.jpg");
Image<BGR> result = Image<BGR>(raw)
    | Resize{}.width(224).height(224)
    | CLAHE{}.clip_limit(2.0)
    | GaussianBlur{}.kernel_size(3);

// 2. Convert for model inference
Image<Float32C3> tensor = result | ToFloat32C3{} | NormalizeTo{0.0f, 1.0f};

// 3. Display a histogram of the processed image
result | Histogram{} | Show{"Histogram"};

// 4. Record from camera with live preview
CameraCapture cam(0);
VideoWriter writer{"output.mp4"};
writer.fps(30);

while (true) {
    auto frame = cam.getFrame();
    if (!frame) continue;
    Image<BGR> img(*frame);
    img | Show{"Live"}.wait_ms(1) | writer;  // display + record in one pipeline
    if (cv::waitKey(1) == 27) break;         // ESC to stop
}
```

## Usage

All ops are functors that compose via `operator|`. The `pipeline.hpp` umbrella header pulls in all core ops. For a full reference of every namespace, op, and error type see **[NAMESPACES.md](NAMESPACES.md)**.

```cpp
// Core ops
#include "improc/core/pipeline.hpp"

// I/O
#include "improc/io/camera_capture.hpp"
#include "improc/io/video_writer.hpp"

// ML utilities
#include "improc/ml/augmentation.hpp"        // all augmentation ops
#include "improc/ml/tracking/tracking.hpp"   // IouTracker, SortTracker, ByteTracker, TrackingEval
#include "improc/ml/dnn_classifier.hpp"
#include "improc/ml/dataset.hpp"

// ONNX Runtime inference
#include "improc/onnx/onnx.hpp"           // OnnxSession, OnnxClassifier, OnnxDetector

// Visualization
#include "improc/visualization/visualization.hpp"

// Lazy views
#include "improc/views/views.hpp"
```

## Augmentation

Stochastic augmentations for training pipelines. Each op takes `(Image<Format>, std::mt19937&)` directly or via `.bind(rng)` for `operator|` use. Format constraints are enforced at compile time via C++20 concepts.

```cpp
#include "improc/ml/augmentation.hpp"
using namespace improc::ml;

std::mt19937 rng(42);

// Single op — direct call
Image<BGR> flipped = RandomFlip{}.p(0.5f)(img, rng);

// Pipeline via .bind()
Image<BGR> result = img
    | RandomFlip{}.p(0.5f).bind(rng)
    | RandomBrightness{}.range(0.8f, 1.2f).bind(rng);

// Full training augmentor
auto augmentor = Compose<BGR>{}
    .add(RandomFlip{}.p(0.5f))
    .add(RandomRotate{}.range(-10.0f, 10.0f))
    .add(RandomApply<BGR>{ColorJitter{}, 0.5f})
    .add(OneOf<BGR>{}
        .add(RandomGaussianNoise{}.std_dev(5.0f, 15.0f))
        .add(RandomSaltAndPepper{}.p(0.02f)));

Image<BGR> augmented = augmentor(img, rng);
// or: Image<BGR> augmented = img | augmentor.bind(rng);
```

See `examples/ml/demo_augmentation.cpp` for a full walkthrough.

### Bbox-Aware Augmentation

All 7 geometric ops also accept `AnnotatedImage<Format>` — ground-truth boxes are transformed alongside the image. Boxes that fall too far outside the frame after the transform are automatically dropped.

```cpp
using namespace improc::ml;

BBox cat{cv::Rect2f(50.f, 50.f, 200.f, 150.f), 0, "cat"};
AnnotatedImage<BGR> ann{img, {cat}};

// Single op — direct call
AnnotatedImage<BGR> result = RandomFlip{}.p(0.5f)(ann, rng);

// BBoxCompose pipeline
BBoxCompose<BGR> pipeline;
pipeline
    .add([](auto a, auto& r){ return RandomFlip{}.p(0.5f)(std::move(a), r); })
    .add([](auto a, auto& r){ return RandomRotate{}.range(-15.f, 15.f)(std::move(a), r); })
    .add([](auto a, auto& r){ return RandomCrop{}.width(224).height(224)(std::move(a), r); });

auto out = pipeline(std::move(ann), rng);
// or: auto out = ann | pipeline.bind(rng);

std::cout << out.boxes.size() << " boxes after augmentation\n";

// Tune the drop threshold per-op (default 0.1 — drop if < 10% visible)
RandomCrop{}.min_area_ratio(0.3f).width(200).height(200)(ann, rng);
```

## Video Recording

```cpp
#include "improc/io/video_writer.hpp"
using improc::io::VideoWriter;

// Basic recording — codec and size auto-detected
VideoWriter writer{"output.mp4"};
writer.fps(30);
writer(frame);            // write a frame
writer.close();           // or let the destructor finalise

// Pipeline form — display and record simultaneously
img | Show{"preview"}.wait_ms(1) | writer;

// Manual configuration
VideoWriter w{"output.avi"};
w.fps(25).size(640, 480).codec("MJPG");
```

Supported auto-codecs: `.mp4`/`.mov` → `mp4v`, `.avi` → `MJPG`, `.mkv` → `XVID`.

## ONNX Runtime Inference

Train models in Python and run inference here. `OnnxClassifier` and `OnnxDetector` share the same fluent API as their `Dnn*` counterparts. `OnnxSession` gives you raw tensor access for custom model architectures.

```cpp
#include "improc/onnx/onnx.hpp"
using namespace improc::onnx;

// Classification — export from PyTorch with torch.onnx.export()
OnnxClassifier cls{"mobilenet.onnx"};
cls.input_size(224, 224)
   .mean(0.485f, 0.456f, 0.406f)
   .scale(1.0f / 255.0f)
   .labels(class_names)
   .top_k(5);

auto results = cls(img);  // std::expected<std::vector<ClassResult>, Error>
if (results)
    for (const auto& r : *results)
        std::cout << r.label << ": " << r.score << "\n";

// Detection — YOLOv8 ONNX export: model.export(format="onnx")
OnnxDetector det{"yolov8n.onnx"};
det.input_size(640, 640).confidence_threshold(0.5f).labels(class_names);

auto boxes = det(frame);  // std::expected<std::vector<Detection>, Error>
if (boxes)
    frame | DrawBoundingBoxes{*boxes} | Show{"Detections"}.wait_ms(1);

// Raw session — custom architectures, multiple outputs
OnnxSession session;
session.load("encoder.onnx");
auto outputs = session.run({{session.input_names()[0], {1,3,224,224}, data}});
```

On **Apple Silicon**, the CoreML execution provider is registered automatically — compatible ops run on the Neural Engine with transparent CPU fallback.

## Edge Detection & Enhancement

```cpp
#include "improc/core/pipeline.hpp"
using namespace improc::core;

// Sobel edge magnitude (accepts Gray or BGR)
Image<Gray> edges = Image<Gray>(gray_mat) | SobelEdge{}.ksize(3);
Image<Gray> edges2 = Image<BGR>(bgr_mat)  | SobelEdge{};  // auto-converts

// Canny (two-threshold hysteresis)
Image<Gray> canny = img | CannyEdge{}.threshold1(50).threshold2(150);

// Gamma correction (any format)
Image<BGR> brighter = img | GammaCorrection{}.gamma(0.5f);  // brighten
Image<BGR> darker   = img | GammaCorrection{}.gamma(2.0f);  // darken

// Bilateral filter (edge-preserving smoothing)
Image<BGR> smooth = img | BilateralFilter{}.diameter(9).sigma_color(75).sigma_space(75);
```

## Drawing Detections

```cpp
#include "improc/visualization/draw.hpp"
using namespace improc::visualization;

// After running DnnDetector
std::vector<improc::ml::Detection> detections = detector(frame);

// Annotate a copy of the frame
Image<BGR> annotated = frame | DrawBoundingBoxes{detections}.thickness(2);

// Suppress labels or confidence scores
Image<BGR> boxes_only = frame | DrawBoundingBoxes{detections}
    .show_label(false).show_confidence(false);
```

## Object Tracking

Three interchangeable trackers, all satisfying `TrackerType<T>`: `IouTracker` (greedy), `SortTracker` (Kalman + Hungarian, SORT), and `ByteTracker` (two-stage BYTE). Pair with `TrackingEval` for offline metric computation.

```cpp
#include "improc/ml/tracking/tracking.hpp"
using namespace improc::ml;

// --- SortTracker — Kalman-filter SORT algorithm ---
SortTracker tracker;
tracker.max_age(3).min_hits(3).iou_threshold(0.3f);

for (auto& frame : video) {
    std::vector<Detection>  dets   = detector(frame);
    std::vector<Track>      tracks = tracker.update(dets);  // confirmed tracks only

    // DrawTracks draws bbox + "ID:N" label — pipeline-compatible
    Image<BGR> annotated = frame | DrawTracks{tracks}.thickness(2);
}

// --- ByteTracker — two-stage matching recovers occluded tracks ---
ByteTracker bt;
bt.max_age(3).min_hits(3)
  .high_conf_threshold(0.6f).low_conf_threshold(0.1f);

// --- Evaluation: MOTA / MOTP / IDF1 / Precision / Recall ---
TrackingEval eval;
eval.iou_threshold(0.5f);

for (int f = 0; f < n_frames; ++f)
    eval.update(predicted_tracks[f], ground_truth[f]);

TrackingMetrics m = eval.compute();
std::cout << "MOTA=" << m.MOTA << "  MOTP=" << m.MOTP
          << "  IDF1=" << m.IDF1
          << "  P="   << m.Precision << "  R=" << m.Recall
          << "  IDSW=" << m.IDSW << "\n";
```

All trackers are drop-in replaceable — swap `SortTracker` for `ByteTracker` with no other changes. `DrawTracks` is in `improc/visualization/draw_tracks.hpp` (included via `visualization.hpp`). See `NAMESPACES.md` for the full setter reference.

<!-- TODO: Realistic tracking demo — requires a short MP4/AVI with trackable objects (people or cars).
     Pipeline: VideoReader → DnnDetector (YOLO 640×640) → ByteTracker → DrawTracks → Show.
     User will provide the video file; wire up examples/ml/demo_tracking_realworld.cpp once available. -->

## Lazy Views

Lazy pipeline adapters over image collections. Nothing executes until you materialise with `views::to<T>()` or a range-for loop.

```cpp
#include "improc/views/views.hpp"
namespace views = improc::views;
using namespace improc::core;

std::vector<Image<BGR>> images = ...;

// transform + filter + take — materialise into a new vector
auto result = images
    | views::filter([](const Image<BGR>& img) { return img.cols() >= 128; })
    | views::transform(Resize{}.width(64).height(64))
    | views::take(10)
    | views::to<std::vector<Image<BGR>>>();

// lazy iteration over a directory — only loads images on demand
for (const auto& img : views::from_dir("dataset/train", {".png", ".jpg"})) {
    // process img ...
}

// batch(N) — iterate over fixed-size chunks
for (const auto& chunk : views::from_dir("frames/", {".png"}) | views::batch(8)) {
    // chunk is std::vector<Image<BGR>> of up to 8 images
}

// enumerate — zero-based index alongside each element
for (const auto& [idx, img] : images | views::take(5) | views::enumerate) {
    std::cout << idx << ": " << img.cols() << "x" << img.rows() << "\n";
}

// zip — pair two sources element-wise (stops at the shorter)
for (const auto& [img, mask] : views::zip(images, masks)) {
    // img and mask aligned by position
}

// compose freely — lazy VideoView source, then batch + enumerate
VideoReader reader{"video.mp4"};
for (const auto& [idx, chunk] : views::VideoView{reader} | views::batch(4) | views::enumerate) {
    std::cout << "Batch " << idx << ": " << chunk.size() << " frames\n";
}
```

See `examples/views/` for the full demo suite (M1–M4) and `NAMESPACES.md` for the complete symbol reference.

## Feature Detection Pipeline

```cpp
#include "improc/core/pipeline.hpp"
using namespace improc::core;

Image<BGR> img1 = ..., img2 = ...;
Image<Gray> gray1 = img1 | ToGray{}, gray2 = img2 | ToGray{};

// Detect → Describe → Match
KeypointSet   kps1  = gray1 | DetectORB{}.max_features(500);
DescriptorSet desc1 = gray1 | DescribeORB{kps1};

KeypointSet   kps2  = gray2 | DetectORB{}.max_features(500);
DescriptorSet desc2 = gray2 | DescribeORB{kps2};

MatchSet ms = MatchBF{desc1, desc2}.cross_check(true)();

// Visualise
Image<BGR> kp_vis    = img1 | DrawKeypoints{kps1};
Image<BGR> match_vis = DrawMatches{img1, kps1, img2, kps2, ms}();

// FLANN + Lowe ratio test (SIFT/float descriptors)
KeypointSet   sift_kps  = gray1 | DetectSIFT{};
DescriptorSet sift_desc = gray1 | DescribeSIFT{sift_kps};
MatchSet      sift_ms   = MatchFlann{sift_desc, sift_desc}.ratio_threshold(0.7f)();
```

## Requirements

- C++23 or later
- [OpenCV](https://opencv.org/) 4.8+
- [ONNX Runtime](https://onnxruntime.ai/) 1.20.1 (auto-downloaded via CMake FetchContent)
- [GoogleTest](https://github.com/google/googletest) 1.16+
- [Conan 2.0](https://conan.io/) for dependency management

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/michalmaj/improc.git
   ```

2. **Configure with CMake (Conan auto-invoked via `conan_provider.cmake`):**
   ```bash
   cmake -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" \
         -DCONAN_COMMAND=<path_to_conan> \
         -B build .
   cmake --build build
   ```

## CMake Configuration (CLion)

Set the following CMake options:
```
-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake"
-DCONAN_COMMAND=<path_to_conan_executable>
```

### Sanitizer Builds (optional)

**Thread Sanitizer:**
```
-DCMAKE_CXX_FLAGS="-fsanitize=thread -g -O1 -fno-omit-frame-pointer"
```

**Address + Undefined Behavior Sanitizer:**
```
-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer"
-DCMAKE_BUILD_TYPE=Debug
```

## Running Tests

```bash
# All tests
cmake --build cmake-build-debug --target improc_tests
./cmake-build-debug/improc_tests

# Single suite
./cmake-build-debug/improc_tests --gtest_filter="CLAHETest.*"
```

## Tested With

- **Compilers:** GCC 14.2, Clang 19.1.7
- **OpenCV:** 4.8.1
- **Eigen:** 3.4.0
- **GoogleTest:** 1.16.0

## Contributing

Contributions are welcome. Fork the repository, create a feature branch (`git checkout -b feature/my-op`), add your changes with tests, and open a pull request against `main`. Please keep new ops consistent with the existing patterns: a fluent-setter functor in `include/improc/`, a matching `.cpp` in `src/`, tests in `tests/`, and the new type registered in the relevant umbrella header (`pipeline.hpp` for core ops, `augmentation.hpp` for augmentation ops). All pull requests must pass the full test suite before merging.
