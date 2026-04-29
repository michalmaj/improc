# Getting Started with improc++

This tutorial gets you from zero to a working image processing pipeline in about 10 minutes.

## Prerequisites

- C++23 compiler: GCC 14+ or Clang 18+
- CMake 3.30+
- Conan 2: `pip install "conan>=2,<3"`
- macOS: Xcode Command Line Tools (`xcode-select --install`)
- Linux: `sudo apt-get install build-essential cmake libgtk-3-dev`

## Install

```bash
git clone https://github.com/michalmaj/improc.git
cd improc
```

## Configure and Build

Conan is invoked automatically by CMake via `conan_provider.cmake`. Point CMake at your Conan binary:

```bash
# Detect your compiler profile (run once per machine)
conan profile detect --force

# Configure — OpenCV, GTest, and Eigen are downloaded and built by Conan
cmake \
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="conan_provider.cmake" \
  -DCONAN_COMMAND=$(which conan) \
  -DCMAKE_BUILD_TYPE=Release \
  -B build .

# Build everything
cmake --build build --parallel
```

## Run the Tests

```bash
./build/improc_tests
```

All tests should pass. The six `VideoWriterTest` cases that show codec warnings are a known environment issue on some systems and do not affect the library.

## Your First Pipeline

Create `hello.cpp`:

```cpp
#include <iostream>
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"

using namespace improc::core;
using namespace improc::io;

int main() {
    // Load an image — imread returns std::expected<Image<BGR>, Error>
    auto src = imread<BGR>("photo.jpg");
    if (!src) {
        std::cerr << "Could not load photo.jpg\n";
        return 1;
    }

    // Build a pipeline: resize → blur → brighten
    Image<BGR> result = *src
        | Resize{}.width(224).height(224)
        | GaussianBlur{}.kernel_size(3)
        | Brightness{}.delta(20.0);

    imwrite("output.jpg", result);
    std::cout << "Saved " << result.cols() << "x" << result.rows() << " image\n";
}
```

Add it to your `CMakeLists.txt`:

```cmake
add_executable(hello hello.cpp)
target_link_libraries(hello PRIVATE improc)
```

Rebuild and run:

```bash
cmake --build build --parallel
./build/hello
```

## What Just Happened

- `imread<BGR>` loads the file and wraps the pixels in an `Image<BGR>`. The format tag is a compile-time guarantee — passing an `Image<Gray>` where `Image<BGR>` is expected is a **compiler error**, not a silent runtime bug.
- Each `|` applies one operation eagerly and returns a new `Image<BGR>`. The source image is never modified.
- `imwrite` encodes and saves the result; the extension determines the format.

## Key Headers

| Header | What it gives you |
|---|---|
| `improc/core/pipeline.hpp` | All core ops (`Resize`, `GaussianBlur`, …) + `operator\|` |
| `improc/io/image_io.hpp` | `imread<F>`, `imwrite` |
| `improc/io/camera_capture.hpp` | Threaded camera capture |
| `improc/io/video_writer.hpp` | RAII video recording |
| `improc/views/views.hpp` | Lazy pipeline adapters |
| `improc/ml/augmentation.hpp` | Stochastic augmentation ops |
| `improc/onnx/onnx.hpp` | ONNX Runtime inference |
| `improc/visualization/visualization.hpp` | Charts, `Show`, `DrawBoundingBoxes` |

## Next Steps

- [Building a Pipeline](building-a-pipeline.md) — composing ops, format conversions, lazy views, augmentation
- [Real-Time Camera](real-time-camera.md) — `CameraCapture` + `ThreadPool` + `FramePipeline`
