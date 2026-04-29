# improc++: Modern C++ Image Processing Toolkit

> **Who is this for?**  
> C++ engineers and researchers building real-time computer vision and ML systems who want
> compile-time safety, composable pipelines, and ML-ready utilities — without leaving the OpenCV ecosystem.

## Status

> **v0.1.0** — first versioned release. APIs are stabilising but may still change between minor versions.

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

- **Not a Python library.** No bindings planned — use OpenCV's Python API for scripting workflows.
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
- **Geometric operations** — `Resize` (aspect-ratio aware), `Crop`, `Flip`, `Rotate`, `Pad`, `PadToSquare`
- **Filter & morphology** — `GaussianBlur`, `MedianBlur`, `Dilate`, `Erode`, `Threshold` (incl. Otsu), `CLAHE`, `GammaCorrection`, `BilateralFilter`
- **Edge detection** — `SobelEdge` (gradient magnitude), `CannyEdge` (hysteresis thresholding); both accept `Image<Gray>` or `Image<BGR>` (auto-converted)
- **Normalization** — `Normalize`, `NormalizeTo`, `Standardize` for ML preprocessing
- **Format conversions** — explicit, compiler-enforced free functions (`convert<Gray>(bgr_image)`)
- **Augmentation** — stochastic training augmentations constrained by C++20 concepts: `RandomFlip`, `RandomRotate`, `RandomCrop`, `RandomResize`, `RandomBrightness`, `RandomContrast`, `ColorJitter`, `RandomGaussianNoise`, `RandomSaltAndPepper`, `Compose`, `RandomApply`, `OneOf`
- **Dataset loading** — load image datasets from class-labeled directories with train/val/test splitting
- **DNN inference** — `DnnClassifier`, `DnnDetector` (YOLO & SSD), `DnnForward` backed by OpenCV DNN; pipeline-composable
- **ONNX Runtime inference** — `OnnxClassifier`, `OnnxDetector` (YOLOv5 & YOLOv8 & SSD), `OnnxSession` (raw tensor I/O); CPU + CoreML EP on Apple Silicon; train in Python, export to ONNX, run here
- **Camera capture** — asynchronous threaded frame capture via `CameraCapture`; `getFrame()` returns `std::expected<cv::Mat, Error>` for safe error handling
- **Video recording** — synchronous RAII `VideoWriter` with auto codec detection and pipeline support (`img | Show{"preview"} | writer`)
- **Haar Cascade loader** — CRTP-based model loader for OpenCV cascade classifiers
- **Threading** — `ThreadPool` and `FramePipeline<Result>` for real-time frame processing
- **Visualization** — `Histogram`, `LinePlot`, `Scatter` chart functors, a `Show` passthrough display op, and `DrawBoundingBoxes` for annotating `Detection` results — all composable with `operator|`
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
#include "improc/ml/augmentation.hpp"     // all augmentation ops
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
