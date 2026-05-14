# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Table of Contents

- [[0.1.0]](#010--2026-04-26) — 2026-04-26 · First versioned release; full namespace surface established
- [[0.2.0]](#020--2026-05-02) — 2026-05-02 · `improc::core` extras + `improc::views` lazy pipeline
- [[0.3.0]](#030--2026-05-07) — 2026-05-07 · Core completeness: morphology, colour spaces, feature detection pipeline
- [[Unreleased]](#unreleased)

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

## [0.2.0] — 2026-05-02

### Added

#### `improc::core`
- **New geometric ops:** `CenterCrop` (center-anchored crop to target size), `LetterBox` (aspect-ratio preserving resize with configurable padding)
- **New threshold op:** `AdaptiveThreshold` (Gaussian and mean block-local thresholding; Gray only; `block_size` must be odd ≥ 3)
- **New pixel ops:** `Invert` (per-channel bitwise NOT for integer formats), `InRange` (binary mask from per-channel lower/upper range bounds)
- **New morphology ops:** `MorphOpen` (Dilate→Erode sequence; removes small foreground blobs), `MorphClose` (Erode→Dilate sequence; fills small holes in foreground)
- **New enhancement ops:** `HistogramEqualization` (contrast normalization via `cv::equalizeHist`; BGR variant operates on Y channel in YCrCb to preserve colour balance), `NLMeansDenoising` (Non-Local Means noise reduction via `cv::fastNlMeansDenoising` / `cv::fastNlMeansDenoisingColored`)
- **New edge detection op:** `LaplacianEdge` (second-derivative edge detector; CV_16S intermediate captures negative responses, `cv::convertScaleAbs` folds to CV_8U; BGR auto-converted to Gray)
- **New concept:** `IntegerFormat` — constrains ops to integer-type image formats

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

---

## [0.3.0] — 2026-05-07

Core Completeness release. `improc::core` now covers the full classical 2D computer vision pipeline:
morphological extras, colour space ops, pyramid ops, annotation drawing, contour analysis,
connected-component labelling, distance transform, and the complete feature detection →
description → matching → visualisation chain.

### Added

#### `improc::core` — Morphology

- **`MorphGradient`** — morphological gradient (dilate − erode); highlights object boundaries; same fluent API as `MorphOpen`/`MorphClose` (`kernel_size`, `shape`)
- **`TopHat`** — white top-hat (source − MorphOpen); isolates small bright features against a dark background
- **`BlackHat`** — black top-hat (MorphClose − source); isolates small dark features against a bright background

#### `improc::core` — Corner Detection

- **`HarrisCorner`** — Harris–Stephens corner detector; returns a float corner-response map normalized to `Image<Gray>`; setters: `block_size` (default 2), `ksize` (Sobel kernel: 3/5/7, default 3), `k` (sensitivity, default 0.04; must be in (0, 1))

#### `improc::core` — Colour Spaces

- **`LAB`** format tag — CIE L\*a\*b\* (CV_8UC3); added to `format_traits.hpp`
- **`YCrCb`** format tag — YCrCb (CV_8UC3); added to `format_traits.hpp`
- **`ToLAB`** — converts `Image<BGR>` → `Image<LAB>` via `cv::COLOR_BGR2Lab`
- **`ToYCrCb`** — converts `Image<BGR>` → `Image<YCrCb>` via `cv::COLOR_BGR2YCrCb`
- **`ToBGR`** — two new overloads: `Image<LAB>` → `Image<BGR>` and `Image<YCrCb>` → `Image<BGR>`

#### `improc::core` — Pyramid Ops

- **`PyrDown`** — Gaussian pyramid downscale to `ceil(rows/2) × ceil(cols/2)` via `cv::pyrDown`; works on any `Image<Format>`
- **`PyrUp`** — Gaussian pyramid upscale to `2*rows × 2*cols` via `cv::pyrUp`; works on any `Image<Format>`

#### `improc::core` — Drawing / Annotation

- **`DrawText`** — renders text on a BGR image clone; setters: `position`, `font_scale` (must be > 0), `color`, `thickness` (must be > 0)
- **`DrawLine`** — draws an antialiased line; setters: `color`, `thickness` (must be > 0)
- **`DrawCircle`** — draws an antialiased circle; `radius` validated at construction (must be > 0); `thickness(-1)` fills
- **`DrawRectangle`** — draws an antialiased rectangle; `thickness(-1)` fills

#### `improc::core` — Contour Analysis

- **`ContourSet`** — result type with `contours` (`std::vector<std::vector<cv::Point>>`), `hierarchy` (`std::vector<cv::Vec4i>`), and bounds-checked accessors `area(i)`, `perimeter(i)`, `bounding_rect(i)`
- **`FindContours`** — extracts contours from a binary `Image<Gray>`; setters: `mode` (External/List/CComp/Tree, default External), `method` (None/Simple/TehChin, default Simple)
- **`DrawContours`** — draws a `ContourSet` onto a BGR image clone; setters: `index` (default −1 = all), `color`, `thickness` (−1 = fill)

#### `improc::core` — Connected Components & Distance Transform

- **`ComponentMap`** — result type with `labels` (CV_32S), `stats` (N×5), `centroids` (N×2), `num_labels`; bounds-checked accessors `area(i)`, `bounding_rect(i)`, `centroid(i)`, `mask(i)`
- **`ConnectedComponents`** — labels connected regions in a binary `Image<Gray>` via `cv::connectedComponentsWithStats`; connectivity setter: `Four` or `Eight` (default Eight)
- **`DistanceTransform`** — distance-to-nearest-zero-pixel for each non-zero pixel; returns `Image<Float32>`; setters: `dist_type` (L1/L2/C, default L2), `mask_size` (Mask3/Mask5/Precise, default Mask3)

#### `improc::core` — Feature Detection Pipeline

- **`KeypointSet`** — result type with `keypoints` (`std::vector<cv::KeyPoint>`), `size()`, `empty()`
- **`DetectORB`** — ORB keypoint detector; setters: `max_features` (default 500, must be > 0), `scale_factor` (default 1.2), `n_levels` (default 8)
- **`DetectSIFT`** — SIFT keypoint detector; setters: `max_features` (default 0 = no limit, must be ≥ 0), `n_octave_layers` (default 3)
- **`DetectAKAZE`** — AKAZE keypoint detector; setter: `threshold` (default 0.001f, must be > 0)
- **`DescriptorSet`** — result type with `KeypointSet keypoints` and `cv::Mat descriptors` (CV_32F for SIFT; CV_8U for ORB/AKAZE); `size()`, `empty()`
- **`DescribeORB`** — computes ORB descriptors (CV_8U, 32 bytes/keypoint); accepts `Image<Gray>` or `Image<BGR>`
- **`DescribeSIFT`** — computes SIFT descriptors (CV_32F, 128 floats/keypoint); accepts `Image<Gray>` or `Image<BGR>`
- **`DescribeAKAZE`** — computes AKAZE descriptors (CV_8U); accepts `Image<Gray>` or `Image<BGR>`
- **`MatchSet`** — result type with `matches` (`std::vector<cv::DMatch>`), `size()`, `empty()`
- **`MatchBF`** — brute-force matcher; norm auto-detected (NORM_HAMMING for CV_8U, NORM_L2 for CV_32F); setters: `cross_check(bool)`, `max_distance(f)` (0 = no filter; must be ≥ 0)
- **`MatchFlann`** — FLANN matcher with Lowe ratio test (`knnMatch k=2`); CV_32F descriptors only (throws `ParameterError` for binary); setter: `ratio_threshold(f)` (default 0.7f; must be in (0, 1])
- **`DrawKeypoints`** — pipeline op; draws keypoints with `DRAW_RICH_KEYPOINTS`; accepts `Image<Gray>` or `Image<BGR>`; always returns `Image<BGR>`
- **`DrawMatches`** — callable (not a pipeline op); renders two BGR images side-by-side with connecting match lines; output width = `img1.cols + img2.cols`

---

## [Unreleased]

### Added

#### `improc::ml`
- **`LabeledImage<F>`** — paired image + soft label type (`std::vector<float>`) for classification augmentation; `operator|` support for pipeline use (`labeled.hpp`)
- **`MixUp`** — blends two `LabeledImage<F>` with λ ~ Beta(α,α); image via `cv::addWeighted`, label as convex combination; setters: `alpha(a)` (> 0; default 0.4), `p(prob)` ([0,1]; default 1.0)
- **`CutMix`** — pastes a random rectangular patch from secondary onto primary; label mixed by actual area ratio 1 − (w·h)/(W·H); setters: `alpha(a)` (> 0; default 1.0), `p(prob)` ([0,1]; default 1.0)
- **`MixCompose<F>`** — sequential composer for binary mix ops; primary passes through each op sequentially with secondary fixed; `bind(secondary, rng)` returns `operator|`-compatible unary functor
- **`VocDataset`** — loads Pascal VOC XML annotation datasets into `AnnotatedImage<BGR>` train/val/test splits; auto-detects VOC split (`ImageSets/Main/`) vs random split; class mapping auto-built or user-supplied via `.classes()`; `skip_difficult` (default true); fluent setter API consistent with `Dataset`
- **`parse_voc_xml`** — free function; parses one VOC XML file + loads image; mutates a shared class map; `filter_unknown=true` drops objects not in the pre-filled map (used internally by `VocDataset`)
- **`CocoDataset`** — loads COCO JSON annotation datasets into `AnnotatedImage<BGR>` splits; explicit `load_train`/`load_val`/`load_test` with shared class mapping; non-contiguous COCO category IDs remapped to 0-indexed sequential IDs; `skip_crowd` (default true); class order user-supplied via `.classes()` (must call before first load)
- **`parse_coco_json`** — free function; parses one COCO JSON file + loads images; mutates a shared class map; `filter_unknown=true` drops objects not in the pre-filled map; requires `nlohmann/json` (via CMake FetchContent)
