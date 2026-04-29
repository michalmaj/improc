# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-04-26

First versioned release. Establishes the full namespace surface and API conventions
that subsequent releases will extend without breaking.

### Added

#### `improc::core`
- `Image<Format>` — compile-time type-safe wrapper over `cv::Mat`; throws on type mismatch or empty mat; shallow-copy semantics with `.clone()` for deep copy
- Format tags: `BGR`, `Gray`, `BGRA`, `Float32`, `Float32C3`, `HSV`; mapped to OpenCV constants via `FormatTraits<F>`
- `convert<To>(img)` — explicit, compiler-enforced free-function format conversions (BGR↔Gray, BGR↔Float32C3, BGR↔HSV, Float32↔Gray, etc.)
- `operator|` pipeline — `img | Resize{}.width(224) | GaussianBlur{}.kernel_size(3)` composition syntax
- C++20 concepts: `AnyFormat`, `BGRFormat`, `GrayFormat`, `MultiChannelFormat`
- Geometric ops: `Resize` (aspect-ratio aware), `Crop`, `Flip`, `Rotate`, `Pad`, `PadToSquare`, `WarpAffine`, `WarpPerspective`, `find_homography`
- Filter ops: `GaussianBlur`, `MedianBlur`, `BilateralFilter`, `UnsharpMask`
- Morphology ops: `Dilate`, `Erode`
- Threshold ops: `Threshold` (Binary, BinaryInv, Truncate, ToZero, Otsu)
- Enhancement ops: `CLAHE`, `GammaCorrection`
- Edge detection: `SobelEdge` (gradient magnitude), `CannyEdge`
- Normalization ops: `Normalize`, `NormalizeTo`, `Standardize`
- Color ops: `Brightness`, `Contrast`, `WeightedBlend`, `AlphaBlend`
- `ApplyMask` — zero-out pixels outside a binary mask
- `pipeline.hpp` umbrella include for all core ops

#### `improc::io`
- `imread<F>(path)` / `imwrite(path, img)` — type-safe file I/O; returns `std::expected<Image<F>, Error>`
- `VideoReader` — sequential frame-by-frame reading with `.next()` → `std::optional<Image<BGR>>`; exposes `width()`, `height()`, `fps()`, `frame_count()`
- `VideoWriter` — RAII video recording with auto codec detection (`.mp4`→`mp4v`, `.avi`→`MJPG`, `.mkv`→`XVID`); pipeline-composable via `operator|`
- `CameraCapture` — asynchronous threaded frame capture; `getFrame()` returns `std::expected<cv::Mat, Error>`
- `io.hpp` umbrella include

#### `improc::ml`
- `ImageLoader` — loads images from a directory into `Image<BGR>` vectors
- `Dataset` — loads class-labelled image datasets with train/val/test splitting
- `ModelLoaderBase<Derived, ModelType>` — CRTP base for OpenCV model loaders (`.yml`/`.yaml`/`.xml`)
- `HaarCascadeLoader` — loads OpenCV Haar cascade classifiers
- `DnnClassifier`, `DnnDetector`, `DnnForward` — inference backed by `cv::dnn`; fluent API consistent with `OnnxClassifier`/`OnnxDetector`
- Augmentation ops (all accept `(Image<F>, std::mt19937&)` or `.bind(rng)` for pipeline use): `RandomFlip`, `RandomRotate`, `RandomCrop`, `RandomResize`, `RandomBrightness`, `RandomContrast`, `ColorJitter`, `RandomGaussianNoise`, `RandomSaltAndPepper`
- Augmentation composers: `Compose<F>`, `RandomApply<F>`, `OneOf<F>`
- `ml.hpp` umbrella include

#### `improc::onnx`
- `OnnxSession` — thin ONNX Runtime 1.20.1 wrapper with pimpl (no ORT headers in public API); CoreML EP auto-registered on Apple Silicon with CPU fallback
- `OnnxClassifier` — full image-to-`ClassResult` pipeline: resize → float → mean subtract → channel swap → HWC→CHW → inference → top-k
- `OnnxDetector` — full image-to-`Detection` pipeline with YOLO (v5/v8 auto-detected) and SSD output parsing, NMS post-processing, and coordinate rescaling
- `onnx.hpp` umbrella include

#### `improc::threading`
- `ThreadPool` — `submit()` returns `std::future<T>`; `submit_detached()` is fire-and-forget; destructor drains queue and joins workers
- `FramePipeline<Result>` — holds references to `CameraCapture` and `ThreadPool`; `tryPop()` returns `std::optional<Result>`

#### `improc::visualization`
- `Histogram`, `LinePlot`, `Scatter` — chart functors; all composable via `operator|`
- `Show` — passthrough display op with configurable `wait_ms`
- `DrawBoundingBoxes` — annotates `Detection` results onto `Image<BGR>`
- `Montage` — arranges a collection of `Image<BGR>` into a configurable grid (cols, cell size, gap, background colour)

#### Infrastructure
- CMake 3.30+ build with Conan 2 dependency management (OpenCV, GTest, Eigen)
- ONNX Runtime 1.20.1 via CMake `FetchContent` (pre-built binary, no source build); opt-in with `-DIMPROC_WITH_ONNX=ON` (default ON)
- Google Benchmark suite; opt-in with `-DIMPROC_BENCHMARKS=ON`
- GitHub Actions CI: macOS (Apple Silicon) + Linux (GCC 14) with ORT binary cache
- Full Doxygen coverage across all public headers
- `NAMESPACES.md` — complete API reference for every namespace, op, error code, and return type

### Changed
- `std::expected<T, std::string>` replaced by `std::expected<T, improc::Error>` with structured error codes throughout `improc::ml` and `improc::onnx`
- Custom exception hierarchy (`improc::ModelError`, `improc::ParameterError`, etc.) replaces raw `std::runtime_error` throws in op constructors

### Requirements
- C++23 (GCC 14+ or Clang 18+)
- CMake 3.30+
- OpenCV 4.8+
- Conan 2.0+ (for local builds)
- ONNX Runtime 1.20.1 (auto-downloaded; requires `-DIMPROC_WITH_ONNX=ON`)

---

## [Unreleased]

### Added

#### `improc::views`
- `views::transform(op)` — lazy single-image and collection transform; defers op execution until materialisation
- `views::filter(pred)` — lazy predicate filter over image collections
- `views::take(n)` / `views::drop(n)` — lazy size-limiting and offset adapters
- `views::to<T>()` — materialisation sink: `to<Image<F>>()` for single images, `to<std::vector<Image<F>>>()` for collections
- `views::from_dir(path, exts)` — lazy directory scanner; images are loaded only as they are iterated
- `views::VideoView{reader}` — lazy frame-by-frame adapter over `VideoReader`
- `views::batch(n)` — groups elements into `std::vector<Image<F>>` chunks of size ≤ n (last chunk may be smaller)
- `views::enumerate` — pairs each element with a zero-based `std::size_t` index; yields `std::pair<std::size_t, Image<F>>`
- `views::zip(v1, v2)` — pairs elements from two sources element-wise; stops at the shorter source
- All adapters compose via `operator|`; `from_dir` and `VideoView` support the full adapter set
- `views.hpp` umbrella include

### Planned
- `improc::cuda` — GPU-accelerated ops via OpenCV CUDA
