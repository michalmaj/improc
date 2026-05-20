# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Table of Contents

- [[Unreleased]](#unreleased) — v0.8.0 in progress (Classic CV ops + pending additions)
- [[0.7.0]](#070--2026-05-19) — 2026-05-19 · Video Pipeline + Packaging: VideoFileCapture, CMake install rules, BackgroundSubtractMOG2/KNN
- [[0.6.0]](#060--2026-05-18) — 2026-05-18 · Real-Time Pipeline: unified camera API (WebcamCapture, IPCameraCapture, OakDCapture), CameraFrame, AnyCameraSource, FramePipeline update
- [[0.5.0]](#050--2026-05-18) — 2026-05-18 · ML Evaluation + Visualization + Multi-Object Tracking; Google Benchmark suite; performance fixes
- [[0.4.0]](#040--2026-05-14) — 2026-05-14 · ML Pipeline: augmentation, dataset loaders (VOC/COCO), segmentation types + seg-aware augmentation + VOC seg loader
- [[0.3.0]](#030--2026-05-07) — 2026-05-07 · Core completeness: morphology, colour spaces, feature detection pipeline
- [[0.2.0]](#020--2026-05-02) — 2026-05-02 · `improc::core` extras + `improc::views` lazy pipeline
- [[0.1.0]](#010--2026-04-26) — 2026-04-26 · First versioned release; full namespace surface established

---

## [Unreleased]

### Added
- `improc::core::LUT` — 256-entry lookup-table pipeline op; applies `cv::LUT` to any `Image<F>` via `operator|`; throws `std::invalid_argument` on wrong table size or depth
- `improc::core::CalcHist` / `CompareHist` — histogram computation and comparison analysis ops; `CalcHist` supports Gray (bins×1) and BGR (3×bins×1 stacked); `CompareHist` wraps `cv::compareHist`
- `improc::core::HoughLinesP` / `HoughCircles` — probabilistic Hough line detection and circle detection analysis ops
- `improc::core::MatchTemplate` — template matching analysis op; returns `{best_match_location, score}`; handles TM_SQDIFF min/max inversion automatically
- `improc::core::Moments` — image moments analysis op; wraps `cv::moments`; `binary` flag for binary images
- `improc::core::Inpaint` — inpainting multi-arg op; TELEA and NS methods; `operator()(img, mask)`
- `improc::core::Watershed` — marker-based segmentation multi-arg op; modifies `cv::Mat& markers` in place
- `improc::core::GrabCut` — foreground/background segmentation multi-arg op; initialized with rect; returns `Image<Gray>` mask
- `improc::core::GoodFeaturesToTrack` — Shi-Tomasi (or Harris) corner detection; returns `std::vector<cv::Point2f>`; fluent: `max_corners()`, `quality_level()`, `min_distance()`, `use_harris()`; throws `ParameterError` on invalid quality/distance
- `improc::core::ConvexHull` — convex hull of a contour (`std::vector<cv::Point>` → `std::vector<cv::Point>`)
- `improc::core::ApproxPolyDP` — Douglas-Peucker polygon approximation; fluent: `epsilon()`, `closed()`
- `improc::core::MinAreaRect` — minimum area bounding rectangle (`std::vector<cv::Point>` → `cv::RotatedRect`)
- `improc::core::BoundingRect` — axis-aligned bounding rectangle (`std::vector<cv::Point>` → `cv::Rect`)
- `improc::core::FloodFill` — flood-fill multi-arg op; BGR and Gray overloads; fluent: `lo_diff()`, `up_diff()`; throws on out-of-bounds seed
- `improc::core::Remap` — general pixel remapping pipeline op; `map1`/`map2` in constructor; fluent: `interpolation()`; composable via `operator|`
- `improc::core::AbsDiff` — per-pixel absolute difference pipeline op; second image in constructor; throws on size/type mismatch
- `improc::core::BitwiseAnd` / `BitwiseOr` — bitwise pipeline ops; second image in constructor; integer formats only; throw on size/type mismatch
- `improc::core::BitwiseNot` — bitwise invert pipeline op (alias for `Invert`); integer formats only
- `improc::core::Flow` — new format tag (CV_32FC2) for dense optical flow fields; follows `FormatTraits` pattern
- `improc::core::SparseLKFlow` — sparse Lucas-Kanade optical flow; tracks `std::vector<cv::Point2f>` across frames; returns `SparseLKFlowResult{points, status, error}`; fluent: `win_size()`, `max_level()`, `max_iter()`, `epsilon()`
- `improc::core::DenseFarnebackFlow` — dense Farneback optical flow; returns `Image<Flow>`; fluent: `pyr_scale()`, `levels()`, `win_size()`, `iterations()`, `poly_n()`, `poly_sigma()`
- `improc::core::DenseDISFlow` — dense DIS optical flow (faster than Farneback); returns `Image<Flow>`; fluent: `preset(UltraFast|Fast|Medium)`
- `improc::core::CamShift` — continuously adaptive MeanShift; takes back-projection + mutable window; returns `CamShiftResult{object, iterations}`; fluent: `epsilon()`, `max_iter()`
- `improc::core::MeanShift` — kernel-based shift; takes back-projection + mutable window; returns iteration count; fluent: `epsilon()`, `max_iter()`
- `improc::core::PhaseCorrelate` — frequency-domain sub-pixel shift estimation; takes two `Image<Float32>`; returns `PhaseCorrelateResult{shift, response}`
- `improc::core::Convolve` — custom 2D convolution pipeline op; kernel in constructor; fluent: `anchor()`, `delta()`, `border()`; throws on empty kernel
- `improc::core::BoxFilter` — averaging (box) blur pipeline op; fluent: `kernel_size()` (default 3), `normalize()` (default true), `border()`
- `improc::core::SobelGradient` — raw Sobel gradients; returns `SobelResult{dx, dy}` (CV_16S); fluent: `ksize()`, `scale()`, `delta()`
- `improc::core::ScharrGradient` — Scharr gradients (more accurate than 3×3 Sobel); returns `ScharrResult{dx, dy}` (CV_16S); fluent: `scale()`, `delta()`
- `improc::core::ConvertScaleAbs` — scale + absolute value → `Image<Gray>` (CV_8U); takes `cv::Mat` directly (for use after Sobel/Laplacian); fluent: `alpha()`, `beta()`
- `improc::core::SplitChannels` — splits `Image<BGR>` → 3×`Image<Gray>`; `Image<BGRA>` → 4×`Image<Gray>`
- `improc::core::MergeChannels` — merges 3 or 4 `Image<Gray>` → `Image<BGR>` or `Image<BGRA>`; throws on size mismatch
- `improc::core::Add` / `Subtract` — element-wise arithmetic pipeline ops; second image in constructor; throw on size/type mismatch
- `improc::core::Multiply` / `Divide` — element-wise arithmetic pipeline ops; second image + optional `scale()`; throw on size/type mismatch; `Divide` by zero follows `cv::divide` semantics (result = 0 for integer types)
- `improc::core::IntegralImage` — summed-area table; returns `IntegralResult{sum, sq_sum}`; fluent: `with_sq_sum(bool)` (default false); output is (rows+1)×(cols+1)
- `improc::core::MinMaxLoc` — finds min/max values and locations; returns `MinMaxLocResult{min_val, max_val, min_loc, max_loc}`; accepts `Image<Gray>` or raw `cv::Mat`
- `improc::core::MeanStdDev` — per-channel mean and standard deviation; returns `MeanStdDevResult{mean, stddev}`; works on any format
- `improc::core::CountNonZero` — count of non-zero pixels; accepts `Image<Gray>`
- `improc::core::Reduce` — reduce image to single row or column; `ReduceOp::{Sum, Avg, Max, Min}`; fluent: `op()`, `dim()` (0=reduce rows, 1=reduce cols)

---

## [0.7.0] — 2026-05-19

### Added
- `improc::io::VideoFileCapture` — reads video files as a `CameraSourceType`; wraps `VideoReader` so any `FramePipeline` works with files identically to live cameras; `Error::EndOfFile` returned at EOF
- `improc::Error::EndOfFile` error code + `Error::end_of_file()` factory
- `improc::core::BackgroundSubtractMOG2` — stateful foreground/background segmentation op using Gaussian Mixture Model; fluent setters: `history()`, `threshold()`, `detect_shadows()`; returns `Image<Gray>` foreground mask
- `improc::core::BackgroundSubtractKNN` — stateful foreground/background segmentation using K-Nearest Neighbours; same interface as MOG2; faster for controlled environments
- CMake install rules: `install(TARGETS improc)`, `improcConfig.cmake`, `improcConfigVersion.cmake` — enables `find_package(improc REQUIRED)` and `target_link_libraries(app PRIVATE improc::improc)` after installation

### Notes
- Background subtractors must be passed as **lvalues** to `operator|` to accumulate state across frames
- CMake packaging is a foundation; Conan Center / vcpkg submission planned for v1.0.0

---

## [0.5.0] — 2026-05-18

ML Evaluation + Visualization release. Adds detection, segmentation, and classification
evaluation accumulators; multi-object tracking (IouTracker, SortTracker, ByteTracker) with
TrackingEval; ML-specific visualizations (confusion matrix, PR/ROC curves, bar charts, IoU
histogram); and a full Google Benchmark suite with public performance documentation.

### Added

#### `improc::ml` — Evaluation

- **`iou()`**, **`average_precision()`** — detection IoU and per-class AP free functions; COCO-style `mAP@0.5` and `mAP@0.5:0.95`
- **`DetectionEval`** — frame-by-frame accumulator; `update(predictions, ground_truth)`; `compute()` → `DetectionMetrics` with `mAP_50`, `mAP_50_95`, per-class `ap_50` map
- **`DetectionEval::pr_curves()`** — per-class sorted `(recall, precision)` pairs for `PRCurvePlot`
- **`pixel_iou()`**, **`dice()`** — per-class IoU and Dice free functions; void pixels (255) ignored
- **`SegEval`** — segmentation accumulator; `compute()` → `SegMetrics` with `per_class_iou`, `per_class_dice`, `mean_iou`, `mean_dice`
- **`accuracy()`**, **`precision_score()`**, **`recall_score()`**, **`f1_score()`** — classification metric free functions
- **`ClassEval`** — classification accumulator with confusion matrix; `compute()` → `ClassMetrics` with per-class P/R/F1 and macro averages

#### `improc::ml` — Multi-Object Tracking

- **`Track`** / **`TrackGT`** — core result and ground-truth annotation types
- **`TrackerType<T>`** — C++20 concept satisfied by all three tracker types; drop-in replaceable
- **`IouTracker`** — greedy IoU matching with age-based culling; no motion model; setters: `min_iou` (default 0.3), `max_age` (default 1)
- **`SortTracker`** — SORT algorithm: constant-velocity Kalman filter + Hungarian assignment on (1 − IoU) cost; setters: `max_age`, `min_hits`, `iou_threshold`
- **`ByteTracker`** — BYTE algorithm: Stage 1 Hungarian on high-confidence detections, Stage 2 greedy IoU on low-confidence detections; setters: `max_age`, `min_hits`, `high_conf_threshold`, `low_conf_threshold`
- **`TrackingEval`** / **`TrackingMetrics`** — MOTA, MOTP, IDF1, Precision, Recall accumulator; IDF1 via global bipartite matching (Hungarian)

#### `improc::visualization` — ML Charts

- **`ConfusionMatrixPlot`** — heatmap from `ClassEval` confusion matrix; normalized per row; colour gradient from white to violet; fluent `.width(int).height(int).title(string)`
- **`PRCurvePlot`** — per-class precision-recall curves from `DetectionEval::pr_curves()`; mAP overlay via `.mAP_50(float)`
- **`ROCCurvePlot`** — per-class ROC curves with AUC annotation; accepts external `fpr_map`/`tpr_map`
- **`ClassBarChart`** — grouped P/R/F1 bars (from `ClassEval`) or single AP bars (from `DetectionEval`); overloaded constructor
- **`IoUHistogram`** — IoU score distribution with configurable bin count and threshold overlay line; setters: `bins(int)`, `threshold(float)`
- **`ml_charts.hpp`** umbrella include for all five ML chart functors
- **`DrawTracks`** — pipeline-composable functor; draws track bboxes with "ID:N" labels on a clone; setters: `color`, `thickness`, `font_scale`, `show_id`

#### Infrastructure

- **Google Benchmark suite** — full per-namespace benchmarks: core pipeline overhead (raw vs improc++ wrapper cost), feature detection, image analysis, ML pipeline, augmentation, eval accumulators, tracking, lazy views (lazy vs eager), and ThreadPool; all ops at two resolutions; opt-in with `-DIMPROC_BENCHMARKS=ON`
- **`BENCHMARKS.md`** — public performance document with quick-reference table, full per-namespace `<details>` tables, and engineering story section (three case studies with before/after data)
- **10 tutorials** in `docs/tutorials/` — ONNX inference, augmentation, evaluation metrics, ML charts, and tracking; plus five gap-fill tutorials for v0.1.0–v0.4.0 features

### Fixed / Performance

- **`IouTracker::update()`** — O(D·T·min(D,T)) → O(D·T·log(D·T)): build IoU matrix once, sort pairs descending, assign greedily in one pass; inner-loop string allocations eliminated. **31× faster at 100 detections** (549 µs → 17.5 µs).
- **`NormalizeTo`**, **`Normalize`**, **`Standardize`** — all six `operator()` overloads (Float32 and Float32C3) now use in-place `convertTo`, eliminating a 600 KB heap allocation per call; ~20% throughput improvement on the ML preprocessing pipeline.

### Documentation

- **`NLMeansDenoising`** — `@warning` Doxygen tag with single-thread times on Apple M4 Pro: 122 ms @ 480×640, up to ~250 ms @ 1080×1920; recommends `GaussianBlur` / `BilateralFilter` for real-time use.
- **`DetectSIFT`**, **`DescribeSIFT`** — `@warning` Doxygen tags with measured times (SIFT detect: 13.6 ms @ 480×640; full pipeline: 311 ms @ 1080×1920); recommends `DetectORB` / `DescribeORB` for real-time use.

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

## [0.6.0] — 2026-05-18

Real-Time Pipeline release. Unified camera API so client code doesn't know what type of camera
it's using. A single `FramePipeline` now accepts webcam, IP camera, or OAK-D depth camera
interchangeably via `AnyCameraSource`. The common currency is `CameraFrame` — a rich frame
type carrying optional depth alongside RGB.

### Added
- `improc::io::CameraFrame` — unified frame type carrying optional RGB (`Image<BGR>`), optional depth (`Image<Float32>`), timestamp, and source ID
- `improc::io::CameraSourceType<T>` — C++20 concept; satisfied by all concrete camera sources
- `improc::io::WebcamCapture` — threaded webcam capture (refactored from `CameraCapture`); `CameraCapture` remains as a backward-compatible alias
- `improc::io::IPCameraCapture` — RTSP/HTTP stream capture via OpenCV, same interface as `WebcamCapture`
- `improc::io::AnyCameraSource` — header-only type-erased camera wrapper for runtime camera selection
- `improc::io::OakDCapture` — OAK-D depth camera support (RGB + metric depth); enabled via `-DIMPROC_WITH_DEPTHAI=ON`; uses depthai-core v2.32.0
- `improc::threading::FramePipeline<Result>` updated: accepts any `CameraSourceType` source (not just `WebcamCapture`); processor function now receives `CameraFrame` instead of `cv::Mat`
- `improc::Error::Timeout` error code + `Error::timeout()` factory for camera queue timeouts
- CMake option `IMPROC_WITH_DEPTHAI` (default `OFF`) for optional OAK-D support via depthai-core v2

### Changed
- `FramePipeline::start()` processor signature changed: `CameraFrame` parameter instead of `cv::Mat` (breaking change, pre-v1.0)

### Notes
- OAK-D integration tests require hardware; run with `./build/improc_tests --gtest_filter="*OakD*"` with device connected via USB3 and `-DIMPROC_WITH_DEPTHAI=ON`

---

## [0.4.0] — 2026-05-14

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
- **`SegmentedImage<F>`** — paired image + class mask (`Image<Gray>`, pixel = class_id, 255 = void kept as-is) + optional instance mask (`std::optional<Image<Gray>>`); `operator|` pipeline support (`segmented.hpp`)
- **Segmentation-aware geometric augmentation** — `RandomFlip`, `RandomRotate`, `RandomCrop`, `RandomResize`, `RandomZoom`, `RandomShear`, `RandomPerspective` gain `SegmentedImage<F>` overloads; masks transformed with `cv::INTER_NEAREST`
- **Segmentation-aware color augmentation** — `RandomBrightness`, `RandomContrast`, `ColorJitter`, `RandomGaussianNoise`, `RandomSaltAndPepper`, `RandomBlur` gain `SegmentedImage<F>` overloads; masks passed through unchanged
- **`SegCompose<F>`** — sequential composer for segmentation augmentation ops; mirrors `BBoxCompose<F>`
- **`VocSegDataset`** — loads Pascal VOC segmentation datasets into `SegmentedImage<BGR>` train/val/test splits; `SegmentationClass/` required, `SegmentationObject/` optional via `load_instance_masks(true)`; VOC split or random 10/10% fallback; `classes()` provides int→string mapping
- **`parse_voc_seg`** — free function; parses one VOC segmentation entry (image + class mask + optional instance mask); supports palette-expanded BGR masks via VOC reverse LUT
- **`BBox`** — annotation type with `cv::Rect2f box`, `int class_id`, `std::string label`
- **`AnnotatedImage<F>`** — paired image + `std::vector<BBox>` for bbox-aware augmentation; `operator|` pipeline support
- **`BBoxCompose<F>`** — sequential composer for bbox-aware augmentation ops; `bind(rng)` returns `operator|`-compatible unary functor; after each transform boxes are clipped and boxes with `clipped_area / original_area < min_area_ratio` (default 0.1) are dropped
- **Bbox-aware geometric overloads** — `RandomFlip`, `RandomRotate`, `RandomCrop`, `RandomResize`, `RandomZoom`, `RandomShear`, `RandomPerspective` gain `AnnotatedImage<F>` overloads; `min_area_ratio` tunable per-op
- **`RandomZoom`** — crops a random sub-region and resizes back to original dimensions; setters: `range(min_scale, max_scale)` — both in (0, 1], min ≤ max (default 0.7, 1.0)
- **`RandomShear`** — affine shear transform; setters: `range(min_deg, max_deg)`, `axis(Axis)` — Horizontal (default) or Vertical
- **`RandomPerspective`** — random homography warp; setter: `distortion_scale(s)` — in [0, 1] (default 0.5)
- **`RandomGrayscale`** — converts BGR to 3-channel grayscale with probability `p` (default 0.1); Gray input unchanged
- **`RandomSolarize`** — inverts pixels at or above a threshold via LUT; setters: `threshold(t)` [0, 255], `p(prob)` [0, 1] (defaults: 128, 0.5)
- **`RandomPosterize`** — reduces bits-per-channel via bitmasking; setters: `bits(b)` [1, 8], `p(prob)` [0, 1] (defaults: 4, 0.5)
- **`RandomEqualize`** — histogram equalization with probability `p`; BGR: operates on Y channel in YCrCb; Gray: direct `cv::equalizeHist`
- **`RandomErasing`** — erases a randomly sampled rectangular region (constant fill); setters: `p`, `scale(min, max)`, `ratio(min, max)`, `value(v)`
- **`GridDropout`** — divides image into cells and independently zeros each with probability `ratio`; setters: `ratio(r)`, `unit_size(s)`, `value(v)`
- **`RandomBlur`** — randomly applies one of Gaussian / Median / Bilateral blur with a random odd kernel size; setters: `types(vector<Type>)`, `kernel_size(min_k, max_k)`
- **`RandomSharpness`** — unsharp-mask sharpening applied with probability `p`; setters: `range(min_s, max_s)`, `p(prob)`
