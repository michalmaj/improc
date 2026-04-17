# improc++: Modern C++ Image Processing Toolkit

## Overview

improc++ is a modern C++23 image processing toolkit designed as a high-level wrapper over OpenCV, aimed at simplifying ML workflows in real-time systems. It provides compile-time type-safe image handling, a composable pipeline API, and structured data loading utilities.

## Features

- **Type-safe image wrapper** — `Image<BGR>`, `Image<Gray>`, `Image<Float32>` etc. catch format mismatches at compile time
- **Composable pipeline** — `image | Resize{}.width(224) | ToFloat32C3{}` syntax for readable processing chains
- **Geometric operations** — `Resize` (aspect-ratio aware), `Crop`, `Flip`, `Rotate`, `Pad`, `PadToSquare`
- **Normalization operations** — `Normalize`, `NormalizeTo`, `Standardize` for ML preprocessing
- **Filter & morphology** — `GaussianBlur`, `MedianBlur`, `Dilate`, `Erode`, `Threshold` (incl. Otsu)
- **Format conversions** — explicit, compiler-enforced free functions (`convert<Gray>(bgr_image)`)
- **Augmentation** — stochastic training augmentations constrained by C++20 concepts: `RandomFlip`, `RandomRotate`, `RandomCrop`, `RandomResize`, `RandomBrightness`, `RandomContrast`, `ColorJitter`, `RandomGaussianNoise`, `RandomSaltAndPepper`, `Compose`, `RandomApply`, `OneOf`
- **Dataset loading** — load image datasets from class-labeled directories with train/val/test splitting
- **DNN inference** — `DnnClassifier`, `DnnDetector` (YOLO & SSD), `DnnForward` backed by OpenCV DNN; pipeline-composable
- **Camera capture** — asynchronous threaded frame capture via `CameraCapture`; `getFrame()` returns `std::expected<cv::Mat, Error>` for safe error handling
- **Video recording** — synchronous RAII `VideoWriter` with auto codec detection and pipeline support (`img | Show{"preview"} | writer`)
- **Haar Cascade loader** — CRTP-based model loader for OpenCV cascade classifiers
- **Threading** — `ThreadPool` and `FramePipeline<Result>` for real-time frame processing
- **Visualization** — `Histogram`, `LinePlot`, `Scatter` chart functors and a `Show` passthrough display op, all returning `Image<BGR>` and composable with `operator|`

## Requirements

- C++23 or later
- [OpenCV](https://opencv.org/) 4.8+
- [Eigen](https://eigen.tuxfamily.org/) 3.4+
- [GoogleTest](https://github.com/google/googletest) 1.16+
- [Conan 2.0](https://conan.io/) for dependency management
- macOS: Accelerate framework (built-in); Linux/Windows: OpenBLAS

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

## Running Tests

```bash
# All tests
cmake --build cmake-build-debug --target improc_tests
./cmake-build-debug/improc_tests

# Single suite
./cmake-build-debug/improc_tests --gtest_filter="VideoWriterTest.*"
```

## Tested With

- **Compilers:** GCC 14.2, Clang 19.1.7
- **OpenCV:** 4.8.1
- **Eigen:** 3.4.0
- **GoogleTest:** 1.16.0
