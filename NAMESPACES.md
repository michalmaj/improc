# Namespace Structure in improc++

The library is organized into modular namespaces under the root `improc` namespace. Each namespace has a single responsibility and maps to a corresponding directory under `include/improc/` and `src/`.

---

## `improc` — Root namespace: exceptions and error values

**Headers:** `improc/exceptions.hpp`, `improc/error.hpp`

### Exception hierarchy (`exceptions.hpp`)

All library exceptions inherit from `improc::Exception`, which inherits from `std::exception`. Catch at the granularity you need.

```
std::exception
└── improc::Exception
    ├── improc::FormatError         — wrong image format (has expected_format / actual_format / context)
    ├── improc::ParameterError      — invalid parameter value (has param_name / constraint / context)
    ├── improc::IoError             — I/O failure
    │   ├── improc::FileNotFoundError  — path does not exist or is not a directory (has path())
    │   └── improc::CameraError        — camera open/read failure (has device_id())
    ├── improc::ModelError          — model load failure (has model_path() / reason())
    ├── improc::DataError           — dataset/loader data problem
    └── improc::AugmentError        — augmentation precondition failure
```

**Rule of thumb:** throw for programming errors (wrong format, invalid parameters, misuse of API); return `std::expected<T, Error>` for environmental errors (file not found, camera unavailable, no images loaded).

### `improc::Error` value type (`error.hpp`)

Used as the error channel in `std::expected<T, improc::Error>` returns throughout the library.

```cpp
struct Error {
    enum class Code {
        NoImages, EmptyDataset, DirectoryNotFound,
        InvalidModelFile, CameraUnavailable, CameraFrameEmpty,
        InsufficientPoints, HomographyFailed,
        ImageReadFailed,      // cv::imread returned empty mat
        ImageWriteFailed,     // cv::imwrite returned false
        OnnxModelLoadFailed,  // ORT failed to parse / load the .onnx file
        OnnxInferenceFailed,  // ORT session Run() returned an error
        OnnxSessionNotLoaded  // run() called before load()
    };
    Code        code;
    std::string message;

    // Named constructors
    static Error no_images(const std::string& dir);
    static Error directory_not_found(const std::string& path);
    static Error invalid_model_file(const std::string& path, const std::string& reason);
    static Error camera_unavailable(int device_id);
    static Error camera_frame_empty(int device_id);
    static Error insufficient_points(std::size_t got);
    static Error homography_failed();
    static Error image_read_failed(const std::string& path);
    static Error image_write_failed(const std::string& path);
    static Error onnx_model_load_failed(const std::string& path, const std::string& reason);
    static Error onnx_inference_failed(const std::string& reason);
    static Error onnx_session_not_loaded();
};
```

Access the human-readable description via `.message`:

```cpp
auto result = loader.get_images();
if (!result)
    std::cerr << result.error().message << '\n';
```

---

## `improc::core` — Type-safe image primitives

**Status: Implemented**

Provides a compile-time type-safe wrapper over `cv::Mat`, format conversion utilities, and a pipeline composition operator.

### Format tag types (`format_traits.hpp`)

Empty structs used as template parameters. `FormatTraits<T>` maps each tag to its OpenCV type constant and channel count.

| Tag | `cv_type` | `channels` | `is_float` |
|---|---|---|---|
| `BGR` | `CV_8UC3` | 3 | `false` |
| `Gray` | `CV_8UC1` | 1 | `false` |
| `BGRA` | `CV_8UC4` | 4 | `false` |
| `Float32` | `CV_32FC1` | 1 | `true` |
| `Float32C3` | `CV_32FC3` | 3 | `true` |
| `HSV` | `CV_8UC3` | 3 | `false` |

**`struct HSV {}`** — format tag for HSV color space (H∈[0,179], S∈[0,255], V∈[0,255]).

All `FormatTraits<F>` specializations include `static constexpr bool is_float` — `false` for BGR/Gray/BGRA/HSV, `true` for Float32/Float32C3.

### `Image<Format>` (`image.hpp`)

Thin wrapper over `cv::Mat` with shallow-copy semantics (same as `cv::Mat`). Constructor validates that the `cv::Mat` is non-empty and its type matches `FormatTraits<Format>::cv_type` — throws `FormatError` on type mismatch, `ParameterError` on empty mat.

```cpp
Image<BGR> img(mat);            // validated at construction
Image<BGR> copy = img;          // shallow copy (shared data)
Image<BGR> owned = img.clone(); // deep copy
cv::Mat& raw = img.mat();       // explicit interop with OpenCV
```

### `convert<To, From>()` (`convert.hpp`)

Free function template. Primary template is `= delete` — unsupported conversions are compile errors. Explicit specializations define allowed conversions:

- `BGR` ↔ `Gray`
- `BGR` ↔ `BGRA`
- `Gray` ↔ `Float32` (scale ×1/255 / ×255)
- `BGR` ↔ `Float32C3` (scale ×1/255 / ×255, 3-channel float)
- `BGR` → `HSV` (OpenCV `COLOR_BGR2HSV`)
- `HSV` → `BGR` (OpenCV `COLOR_HSV2BGR`)

```cpp
Image<Gray>     gray  = convert<Gray, BGR>(bgr_image);
Image<Float32C3> f32c3 = convert<Float32C3, BGR>(bgr_image);
```

### Pipeline `operator|` and ops (`pipeline.hpp`)

Composes processing steps using the pipe operator. Each step is a small functor. `pipeline.hpp` is the single umbrella include — it pulls in all ops.

```cpp
// Conversion functors
Image<Float32>   result = bgr_image | ToGray{} | ToFloat32{};
Image<Float32C3> result = bgr_image | ToFloat32C3{};
Image<HSV>       hsv    = bgr_image | ToHSV{};
Image<BGR>       bgr    = hsv_image | ToBGR{};  // also accepts Image<Gray> (existing overload)

// Geometric ops — templated, work on any Image<Format>
Image<BGR> r = src | Resize{}.width(224).height(224);   // both dims
Image<BGR> r = src | Resize{}.width(224);               // aspect-ratio preserved
Image<BGR> c = src | Crop{}.x(10).y(10).width(100).height(100);
Image<BGR> f = src | Flip{Axis::Horizontal};            // Axis: Horizontal/Vertical/Both
Image<BGR> t = src | Rotate{}.angle(30.0).scale(0.5);  // scale optional, default 1.0

// Normalization ops — Image<Float32> only
Image<Float32> n  = f32 | Normalize{};                  // auto min-max → [0, 1]
Image<Float32> n2 = f32 | NormalizeTo{-1.0f, 1.0f};    // explicit target range
Image<Float32> n3 = f32 | Standardize{0.485f, 0.229f}; // z-score

// Blur ops — templated, work on any Image<Format>
Image<BGR>  blurred = src  | GaussianBlur{}.kernel_size(5).sigma(1.0);
Image<Gray> median  = gray | MedianBlur{}.kernel_size(3);

// Morphological ops — templated, work on any Image<Format>
Image<Gray> dilated = mask | Dilate{}.kernel_size(5).shape(MorphShape::Ellipse);
Image<Gray> eroded  = mask | Erode{}.kernel_size(3).iterations(2);

// Threshold — templated; Otsu only valid on Image<Gray> (CV_8U)
Image<Gray> binary = gray | Threshold{}.value(128).mode(ThresholdMode::Binary);
Image<Gray> otsu   = gray | Threshold{}.mode(ThresholdMode::Otsu);

// Padding ops — templated, work on any Image<Format>
Image<BGR>  padded = src | Pad{}.top(10).bottom(10).left(20).right(20).mode(PadMode::Reflect);
Image<BGR>  square = src | PadToSquare{}.value({114, 114, 114});  // letterbox for inference

// CLAHE — Image<Gray> (direct) or Image<BGR> (applied to L channel in LAB)
Image<Gray> eq  = gray | CLAHE{};                                  // defaults: clip=40, tile=8×8
Image<Gray> eq2 = gray | CLAHE{}.clip_limit(2.0).tile_grid_size(8, 8);
Image<BGR>  eq3 = bgr  | CLAHE{}.clip_limit(3.0);                 // colour-safe via LAB
```

**Geometric ops** throw `ParameterError` when required parameters are missing or invalid (e.g. no dimension in `Resize`, ROI out of bounds in `Crop`, no angle in `Rotate`).

**Normalization ops** throw `ParameterError` at construction if parameters are invalid (`NormalizeTo` requires `min < max`; `Standardize` requires `std_dev > 0`). A uniform image passed to `Normalize` or `NormalizeTo` returns a zero-filled image.

**`CLAHE`** throws `ParameterError` for non-positive `clip_limit` or tile dimensions. On BGR input the operation is colour-safe: it converts to LAB, equalises the L channel only, and converts back — so hue and saturation are preserved.

```cpp
// GammaCorrection — any format (Gray, BGR, Float32, Float32C3)
// output = input ^ gamma  (gamma > 1 darkens, gamma < 1 brightens)
Image<BGR>  bright = img | GammaCorrection{}.gamma(0.5f);
Image<Gray> dark   = gray | GammaCorrection{}.gamma(2.0f);
Image<Float32C3> fp = fp_img | GammaCorrection{}.gamma(1.8f);

// BilateralFilter — edge-preserving smoothing (Gray or BGR, 8-bit)
Image<BGR>  smooth = img | BilateralFilter{}.diameter(9).sigma_color(75).sigma_space(75);
Image<Gray> s2     = gray | BilateralFilter{};  // defaults: d=9, sigma_color=75, sigma_space=75

// SobelEdge — gradient magnitude; accepts Gray or BGR (auto-converts to Gray)
// ksize must be 1, 3, 5, or 7
Image<Gray> sob  = gray | SobelEdge{}.ksize(3);
Image<Gray> sob2 = bgr  | SobelEdge{};  // BGR auto-converted to Gray before processing

// CannyEdge — hysteresis thresholding; accepts Gray or BGR
// aperture_size must be 3, 5, or 7
Image<Gray> canny = gray | CannyEdge{}.threshold1(50).threshold2(150);
Image<Gray> c2    = bgr  | CannyEdge{}.threshold1(100).threshold2(200).aperture_size(3);

// WarpPerspective — apply a 3×3 homography to an image
Image<BGR> warped = src | WarpPerspective{}.homography(H).width(640).height(480);
```

**`GammaCorrection`** throws `ParameterError` if `gamma <= 0`. Uses an 8-bit LUT for integer formats (fast); `cv::pow` + clamp for float formats.

**`BilateralFilter`** throws `ParameterError` for non-positive `diameter`, `sigma_color`, or `sigma_space`. Supports `Gray` and `BGR` only (OpenCV bilateral requires 8-bit input).

**`SobelEdge`** throws `ParameterError` if `ksize` is not in {1, 3, 5, 7}. Computes X and Y gradients in CV_32F, combines via `cv::magnitude`, and converts back to CV_8U.

**`CannyEdge`** throws `ParameterError` for negative thresholds or invalid `aperture_size` (not in {3, 5, 7}).

**`find_homography(src, dst, threshold)`** — free function; computes a 3×3 homography matrix from ≥4 corresponding point pairs via RANSAC. Returns `std::expected<cv::Mat, Error>` — error if fewer than 4 points are provided or RANSAC fails to find a valid homography.

| Function / Op | Return type | Description |
|---|---|---|
| `find_homography(src, dst, threshold)` | `std::expected<cv::Mat, Error>` | Compute 3×3 homography from ≥4 point pairs via RANSAC |
| `WarpPerspective` | pipeline op | Apply homography; `.homography(H)`, `.width(w)`, `.height(h)` |
| `WarpAffine` | pipeline op | Apply a 2×3 affine transformation; `.matrix(M)`, `.width(w)`, `.height(h)` |
| `ApplyMask` | pipeline op | Zero pixels outside a binary `Image<Gray>` mask; `.mask(m)` |
| `UnsharpMask` | pipeline op | Sharpen via blur subtraction; `.sigma(s)`, `.strength(f)` |

**`WarpPerspective`** throws `ParameterError` if `.homography()` is not set, or if width/height are not positive.

```cpp
// Brightness and Contrast — work on any Image<Format>
Image<BGR>  bright   = img  | Brightness{}.delta(30.0);   // additive; clamped to valid range
Image<Gray> dark     = gray | Brightness{}.delta(-20.0);
Image<BGR>  contrast = img  | Contrast{}.factor(1.5);     // multiplicative

// WeightedBlend — blends two same-format, same-size images
Image<BGR>  blended  = img  | WeightedBlend<BGR>{img2}.alpha(0.6);  // out = α·img + (1−α)·img2

// AlphaBlend — composites a BGRA overlay onto a BGR background
Image<BGR>  composited = bg | AlphaBlend{overlay_bgra};   // per-pixel alpha from overlay's A channel
```

**`Brightness`** — additive brightness adjustment via `.delta(double)`. Works on any `AnyFormat`. Pixel values are clamped to the valid range for the format.

**`Contrast`** — multiplicative contrast scaling via `.factor(double)`. Works on any `AnyFormat`. Throws `ParameterError` if `factor <= 0`.

**`WeightedBlend<F>`** — weighted average of two same-format images. `.alpha(double)` controls the blend weight of the first image (second image weight = 1−alpha). Throws `ParameterError` if `alpha` is outside `[0, 1]` or if image sizes differ.

**`AlphaBlend`** — composites a BGRA overlay image onto a BGR background using per-pixel alpha from the overlay's A channel. Throws `ParameterError` if overlay and background sizes differ.

---

## `improc::io` — Input/Output

**Status: Implemented**

### `imread<F>` / `imwrite` (`io/image_io.hpp`)

Typed image I/O functions. Both return `std::expected` for error handling — no exceptions thrown for environmental failures.

```cpp
#include "improc/io/image_io.hpp"
using namespace improc::io;

// Read and auto-convert to the requested format
auto result = imread<BGR>("/path/to/image.png");
if (!result)
    std::cerr << result.error().message << '\n';
else
    Image<BGR> img = *result;

// Read directly as Gray or Float32C3
auto gray = imread<Gray>("/path/to/image.png");
auto fp   = imread<Float32C3>("/path/to/image.png");

// Write any Image<F> to file (format inferred from extension)
auto ok = imwrite("/path/to/output.png", img);
if (!ok)
    std::cerr << ok.error().message << '\n';
```

`imread<F>` calls `cv::imread` then `convert<F, BGR>()` to produce the requested format. Returns `Error{Code::ImageReadFailed}` if `cv::imread` returns an empty mat.

`imwrite` calls `cv::imwrite`. Returns `Error{Code::ImageWriteFailed}` if `cv::imwrite` returns false (bad path, unsupported extension, permissions error).

### `CameraCapture` (`io/camera_capture.hpp`)

Asynchronous camera frame capture. Runs a background capture thread. Non-copyable, non-movable.

`getFrame()` returns `std::expected<cv::Mat, improc::Error>` — use `.has_value()` / `*result` / `.error()` to inspect the result.

```cpp
CameraCapture cam(0);
std::this_thread::sleep_for(std::chrono::milliseconds(500)); // warm-up

if (auto frame = cam.getFrame())
    cv::imshow("Live", *frame);
else
    std::cerr << frame.error().message << '\n';

cam.stop();
```

### `VideoWriter` (`io/video_writer.hpp`)

Synchronous RAII video writer. Size is auto-detected from the first frame if not set explicitly. Codec is auto-detected from the file extension. `operator()` writes a frame and returns it unchanged — pipeline-compatible.

| Extension | Auto codec |
|---|---|
| `.mp4`, `.mov`, `.m4v` | `mp4v` |
| `.avi` | `MJPG` |
| `.mkv` | `XVID` |

```cpp
// Basic usage — RAII, destructor finalises the file
{
    VideoWriter w{"output.mp4"};
    w.fps(30);
    for (auto& frame : frames)
        w(frame);
}  // file closed here

// Pipeline form — display and record at the same time
img | Show{"preview"}.wait_ms(1) | writer;

// Explicit configuration
VideoWriter w{"output.avi"};
w.fps(25).size(640, 480).codec("MJPG");
```

Throws `ParameterError` on invalid setter arguments; throws `IoError` if the underlying `cv::VideoWriter` fails to open or if a frame's size does not match the writer's configured size.

### `VideoReader` (`io/video_reader.hpp`)

Sequential video file reader. `next()` returns `std::optional<Image<BGR>>` — reads frames one by one until EOF, then returns `std::nullopt`.

---

## `improc::ml` — Machine Learning utilities

**Status: Implemented**

### `ImageLoader`

Loads all valid images (`.jpg`, `.jpeg`, `.png`) from a directory. `load_images()` throws `FileNotFoundError` if the path is not a directory. `get_images()` returns `std::expected<std::vector<cv::Mat>, improc::Error>` — error code `NoImages` if nothing was loaded.

### `Dataset`

Loads an image dataset from a directory of class-labeled subdirectories. Splits into train/val/test sets with configurable ratios and optional per-class cap.

```
root_dir/
  class_a/
    img1.jpg
  class_b/
    img2.png
```

### `ModelLoaderBase<Derived, ModelType>` (CRTP)

Base class for model loaders. Validates file path and extension (`.yml`, `.yaml`, `.xml`). Concrete loaders implement `load_impl()` and `get_impl()`.

### `HaarCascadeLoader`

Loads OpenCV Haar Cascade XML files into `cv::CascadeClassifier`.

### DNN Inference (`dnn_classifier.hpp`, `dnn_detector.hpp`, `dnn_forward.hpp`)

Standalone inference functors backed by OpenCV DNN. All accept `const Image<BGR>&` and work as both standalone functors and terminal `operator|` pipeline ops. Model is loaded at construction via `cv::dnn::readNet` — throws `ModelError` on failure (file not found, unsupported extension, parse error, empty net). Supported formats: `.onnx`, `.pb`, `.caffemodel`, `.weights`, `.t7`, `.net`.

Shared result types are in `result_types.hpp`: `ClassResult{class_id, score, label}` and `Detection{box, class_id, confidence, label}`.

```cpp
// Classification — top-K results
auto results = img | Resize{}.width(224) | DnnClassifier{"resnet50.onnx"}.top_k(5);
for (const auto& r : results)
    std::cout << r.class_id << " " << r.score << " " << r.label << "\n";

// Detection — YOLO style (default) with NMS
auto detections = DnnDetector{"yolov8n.onnx"}
    .confidence_threshold(0.5f).labels(class_names)(frame);

// Detection — SSD style (two output blobs)
auto detections = DnnDetector{"ssd.onnx"}
    .style(DnnDetector::Style::SSD)
    .boxes_layer("detection_boxes")
    .scores_layer("detection_scores")(frame);

// Raw output tensor (for custom parsing)
std::vector<float> blob = DnnForward{"encoder.onnx"}(img);
```

**`DnnClassifier`** — runs `blobFromImage` → `forward` → `sortIdx`, returns top-K `ClassResult` sorted by score. Setters: `top_k(int)`, `input_size(int,int)`, `mean(cv::Scalar)`, `scale(float)`, `swap_rb(bool)`, `labels(vector<string>)`. Defaults: top_k=5, 224×224, scale=1/255, swap_rb=true.

**`DnnDetector`** — supports two output formats via `Style` enum. `Style::YOLO` (default): single blob `[1, N, 5+C]` (cx,cy,w,h,obj_conf,class_scores). `Style::SSD`: two blobs — boxes `[1,N,4]` as y1,x1,y2,x2 normalized + scores `[1,N,C]`. NMS applied via `cv::dnn::NMSBoxes`. Box coordinates are scaled back to original image dimensions. Defaults: 640×640, confidence_threshold=0.5, nms_threshold=0.4.

**`DnnForward`** — minimal wrapper: `blobFromImage` → `forward` → flatten to `std::vector<float>`. Use for models with non-standard output formats that require custom parsing. Defaults: 224×224, scale=1/255, swap_rb=true.

### Augmentation (`augmentation.hpp`)

Stochastic image augmentation ops for training data pipelines. All ops are header-only and templated on `Image<Format>`. Each op exposes two interfaces:
- `aug(img, rng)` — direct call with caller-owned `std::mt19937`
- `aug.bind(rng)` — returns a functor compatible with `operator|`

Format constraints are enforced at compile time via C++20 concepts (`AnyFormat`, `BGRFormat`).

```cpp
#include "improc/ml/augmentation.hpp"
using namespace improc::ml;
using namespace improc::core;

std::mt19937 rng(42);

// Standalone call
Image<BGR> flipped = RandomFlip{}.p(0.5f)(img, rng);

// Pipeline via .bind()
Image<BGR> result = img
    | RandomFlip{}.p(0.5f).bind(rng)
    | RandomBrightness{}.range(0.8f, 1.2f).bind(rng);

// Composition pipeline for training
auto augmentor = Compose<BGR>{}
    .add(RandomFlip{}.p(0.5f))
    .add(RandomRotate{}.range(-10.0f, 10.0f))
    .add(RandomApply<BGR>{ColorJitter{}.brightness(0.8f, 1.2f), 0.5f})
    .add(OneOf<BGR>{}
        .add(RandomGaussianNoise{}.std_dev(5.0f, 15.0f))
        .add(RandomSaltAndPepper{}.p(0.02f)));

Image<BGR> augmented = augmentor(img, rng);
```

**Geometric ops** (`RandomFlip`, `RandomRotate`, `RandomCrop`, `RandomResize`) — work on any `Image<Format>`. `RandomCrop` throws `ParameterError` if crop exceeds image size or dimensions not set.

**Colour ops** (`RandomBrightness`, `RandomContrast`) — work on any `Image<Format>`. `ColorJitter` requires `Image<BGR>` (compile error on other formats).

**Noise ops** (`RandomGaussianNoise`, `RandomSaltAndPepper`) — work on any `Image<Format>`. Clamp range adapts to pixel depth: `[0, 255]` for 8-bit, `[0, 1]` for float.

**Composition ops** (`Compose<F>`, `RandomApply<F>`, `OneOf<F>`) — parameterised on `Format`, use `std::function` for type erasure. `OneOf` throws `AugmentError` if called with no augmentations added. `RandomApply` throws `ParameterError` if `p` is outside `[0, 1]`.

---

## `improc::threading` — Concurrency utilities

**Status: Implemented**

### `ThreadPool` (`threading/thread_pool.hpp`)

General-purpose thread pool. Workers drain a task queue; `submit` returns a typed future; `submit_detached` is fire-and-forget. Destructor drains the queue and joins all workers (graceful shutdown). Non-copyable, non-movable.

```cpp
ThreadPool pool(4);                                          // 4 workers (default: hardware_concurrency)

auto future = pool.submit([](int a, int b){ return a+b; }, 3, 4);
int result  = future.get();                                  // 7

pool.submit_detached([]{ heavy_work(); });                   // fire-and-forget
```

Throws `ParameterError` if constructed with 0 threads. Exceptions from tasks propagate through `future.get()`.

### `FramePipeline<Result>` (`threading/frame_pipeline.hpp`)

Header-only template. Connects `CameraCapture` to `ThreadPool` for typed frame-by-frame processing. Holds references (not ownership) to both. Non-copyable, non-movable.

```cpp
improc::io::CameraCapture camera(0);
ThreadPool pool(4);
FramePipeline<cv::Mat> pipeline(camera, pool);

pipeline.start([](cv::Mat frame) {
    return frame | Resize{}.width(224).height(224) | ToFloat32C3{};
});

while (running) {
    if (auto result = pipeline.tryPop()) {
        // use *result (cv::Mat)
    }
}
pipeline.stop();
```

`tryPop()` returns `std::optional<Result>` — `std::nullopt` if no result is ready or `start()` was not called. `start()` called twice throws `improc::Exception`. `stop()` is idempotent.

`FramePipeline<void>` is not supported (`std::optional<void>` is not valid C++). For fire-and-forget frame processing use `pool.submit_detached(processor, camera.getFrame())` directly.

---

## `improc::visualization` — Chart and display utilities

**Status: Implemented**

All chart functors return `Image<BGR>` and compose with the existing `operator|` pipeline. No external plotting library — everything uses OpenCV.

### `Histogram` (`visualization/histogram.hpp`)

Pipeline op. Computes per-channel histograms and renders them as curves on a black canvas. Accepts `Image<BGR>`, `Image<Gray>`, or `Image<Float32>`.

```cpp
#include "improc/visualization/visualization.hpp"
using namespace improc::visualization;

Image<BGR> hist  = bgr_image | Histogram{};                          // 512×256, 256 bins
Image<BGR> hist2 = gray_image | Histogram{}.bins(64).width(400).height(200);
Image<BGR> hist3 = f32_image  | Histogram{};                         // auto min-max range
```

BGR input renders three colored curves (blue, green, red). Gray/Float32 input renders a single white curve.

### `LinePlot` (`visualization/line_plot.hpp`)

Standalone functor. Takes a `std::vector<float>` and renders a connected polyline scaled to the canvas.

```cpp
std::vector<float> loss = {1.2f, 0.9f, 0.7f, 0.5f, 0.3f};
Image<BGR> plot = LinePlot{}.title("Train Loss").color({0, 200, 255})(loss);
```

Values are min-max normalised to canvas height. A single value or all-equal values renders a horizontal centre line. Throws `ParameterError` if vector is empty.

### `Scatter` (`visualization/scatter.hpp`)

Standalone functor. Takes `xs` and `ys` vectors and renders filled circles, each axis independently normalised.

```cpp
Image<BGR> sc = Scatter{}.title("Features").color({0, 255, 255}).point_radius(5)(xs, ys);
```

Throws `ParameterError` if either vector is empty or `xs.size() != ys.size()`. An 8 px margin ensures extreme points are fully visible.

### `Show` (`visualization/show.hpp`, header-only)

Passthrough display op. Calls `cv::imshow()` + `cv::waitKey()` and returns the image unchanged — enables inline display anywhere in a pipeline.

```cpp
// Block until key press (default)
bgr_image | Histogram{} | Show{"Histogram"};

// Non-blocking — suitable for camera loops
result | Show{"Camera"}.wait_ms(1);

// Stand-alone chart display
LinePlot{}.title("Loss")(loss) | Show{"Loss"};
```

`Show` accepts only `Image<BGR>`. To display Gray/Float32 images convert first or pipe through `Histogram{}`.

### `DrawBoundingBoxes` (`visualization/draw.hpp`, header-only)

Draws detection boxes (and optional labels / confidence scores) onto a clone of the source image. Takes a `std::vector<improc::ml::Detection>` at construction and returns `Image<BGR>` — pipeline-compatible.

```cpp
#include "improc/visualization/draw.hpp"
using namespace improc::visualization;

std::vector<improc::ml::Detection> dets = detector(frame);

// Annotate with default settings (green boxes, thickness 2, label + confidence)
Image<BGR> annotated = frame | DrawBoundingBoxes{dets};

// Custom appearance
Image<BGR> annotated = frame | DrawBoundingBoxes{dets}
    .color({0, 0, 255})      // red
    .thickness(1)
    .font_scale(0.4)
    .show_label(true)
    .show_confidence(false);

// Full inference + annotation pipeline
Image<BGR> result = frame
    | Resize{}.width(640).height(640)
    | DrawBoundingBoxes{detector(frame)}.thickness(2)
    | Show{"Detections"}.wait_ms(1);
```

`DrawBoundingBoxes` draws onto a **clone** — the source image is never modified. Throws `ParameterError` if `thickness <= 0` or `font_scale <= 0`.

### `Montage` (`visualization/montage.hpp`)

Arranges a `vector<Image<BGR>>` into a grid. Setters: `.cols(int)`, `.cell_size(int w, int h)`, `.gap(int)`, `.background(cv::Scalar)`. Returns a single tiled `Image<BGR>`.

### Umbrella include

```cpp
#include "improc/visualization/visualization.hpp"  // includes all five
```

---

---

## `improc::onnx` — ONNX Runtime inference

**Status: Implemented** (ONNX Runtime 1.20.1 — CPU + CoreML EP on Apple Silicon)

**Umbrella header:** `improc/onnx/onnx.hpp`

Downloaded automatically via CMake FetchContent — no separate installation required. On Apple Silicon the CoreML execution provider is registered for compatible ops with transparent CPU fallback.

### `TensorInfo` (`onnx_session.hpp`)

Exchange type between `OnnxSession` and its callers. Carries name, shape, and flat float data.

```cpp
struct TensorInfo {
    std::string          name;   // input/output node name
    std::vector<int64_t> shape;  // e.g. {1, 3, 224, 224}
    std::vector<float>   data;   // flat row-major floats
};
```

### `OnnxSession` (`onnx_session.hpp`)

Thin wrapper over `Ort::Session`. ORT types are fully hidden behind a pimpl — including `onnx_session.hpp` does not require `onnxruntime_cxx_api.h` in caller code.

```cpp
OnnxSession session;
auto err = session.load("model.onnx");  // std::expected<void, Error>
if (!err) { std::cerr << err.error().message; }

auto outputs = session.run({{session.input_names()[0], {1,3,224,224}, data}});
// returns std::expected<std::vector<TensorInfo>, Error>

session.input_names();   // std::vector<std::string>
session.output_names();  // std::vector<std::string>
session.is_loaded();     // bool
```

`load()` returns `Error{Code::InvalidModelFile}` for a missing file or wrong extension; `Error{Code::OnnxModelLoadFailed}` if ORT rejects the model. `run()` returns `Error{Code::OnnxSessionNotLoaded}` if called before `load()`, or `Error{Code::OnnxInferenceFailed}` on ORT error.

`OnnxSession` is **non-copyable, movable**.

### `OnnxClassifier` (`onnx_classifier.hpp`)

Wraps `OnnxSession` with a complete image preprocessing pipeline and top-K post-processing. Mirrors the `DnnClassifier` fluent API.

```cpp
OnnxClassifier cls{"mobilenet.onnx"};
cls.input_size(224, 224)
   .mean(0.485f, 0.456f, 0.406f)   // B, G, R order
   .scale(1.0f / 255.0f)
   .swap_rb(true)                   // BGR → RGB before inference
   .labels(class_names)
   .top_k(5);

auto result = cls(img);  // std::expected<std::vector<ClassResult>, Error>
```

Expects the model's first output to be a 1-D score vector `[1, N]`. Results are sorted by score descending. Constructor throws `ModelError` if the model cannot be loaded.

| Setter | Default | Throws |
|---|---|---|
| `top_k(int k)` | 5 | `ParameterError` if k ≤ 0 |
| `input_size(int w, int h)` | 224 × 224 | `ParameterError` if either ≤ 0 |
| `mean(float b, float g, float r)` | 0, 0, 0 | — |
| `scale(float s)` | 1/255 | `ParameterError` if s ≤ 0 |
| `swap_rb(bool)` | `true` | — |
| `labels(vector<string>)` | empty | — |

### `OnnxDetector` (`onnx_detector.hpp`)

Wraps `OnnxSession` with image preprocessing, format-aware YOLO/SSD output parsing, and NMS. Mirrors the `DnnDetector` fluent API.

```cpp
OnnxDetector det{"yolov8n.onnx"};
det.input_size(640, 640)
   .confidence_threshold(0.5f)
   .nms_threshold(0.4f)
   .labels(class_names);

auto boxes = det(frame);  // std::expected<std::vector<Detection>, Error>
```

Two output formats are supported via `Style`:

| Style | Output layout | Notes |
|---|---|---|
| `Style::YOLO` (default) | `[1, N, 5+C]` (YOLOv5) or `[1, 4+C, N]` (YOLOv8) | Auto-detected by shape heuristic |
| `Style::SSD` | boxes `[1,N,4]` (y1,x1,y2,x2 norm.) + scores `[1,N,C]` | Two output tensors required |

YOLOv8 ONNX export via `model.export(format="onnx")` produces `[1, 4+C, N]` — handled automatically. Box coordinates are rescaled to the original image dimensions before NMS.

| Setter | Default | Throws |
|---|---|---|
| `style(Style)` | `YOLO` | — |
| `confidence_threshold(float)` | 0.5 | `ParameterError` if outside [0,1] |
| `nms_threshold(float)` | 0.4 | `ParameterError` if outside [0,1] |
| `input_size(int w, int h)` | 640 × 640 | `ParameterError` if either ≤ 0 |
| `mean(float b, float g, float r)` | 0, 0, 0 | — |
| `scale(float s)` | 1/255 | `ParameterError` if s ≤ 0 |
| `swap_rb(bool)` | `true` | — |
| `labels(vector<string>)` | empty | — |

---

## `improc::views` — Lazy Image Pipeline

**Header:** `#include "improc/views/views.hpp"`

Lazy, ranges-style pipeline layer. Unlike the eager `operator|` in `improc::core`,
adapters from `improc::views` build a deferred computation chain that executes only
when `views::to<T>()` is called — avoiding intermediate `Image<F>` copies.

### M1 — Single Image

| Symbol | Description |
|---|---|
| `views::transform(op)` | Wraps any `improc::core` op into a lazy adapter; no computation at this point |
| `views::to<Image<F>>()` | Materializes the lazy chain into `Image<F>`; triggers the full op sequence |

Format mismatch between the view and `to<>()` is a **compile error** — no runtime check needed.

**Example:**
```cpp
#include "improc/views/views.hpp"
namespace views = improc::views;
using namespace improc::core;

auto view = img
    | views::transform(Resize{}.width(224).height(224))
    | views::transform(GaussianBlur{}.kernel_size(3))
    | views::transform(Brightness{}.delta(10.0));

Image<BGR> result = view | views::to<Image<BGR>>();
```

### M2 — In-Memory Collections (`std::vector<Image<F>>`)

| Symbol | Description |
|---|---|
| `views::transform(op)` | Applies op lazily to each collection element |
| `views::filter(pred)` | Skips elements where `pred(img)` returns `false` |
| `views::take(n)` | Stops iteration after `n` elements |
| `views::drop(n)` | Skips the first `n` elements |
| `views::to<std::vector<Image<F>>>()` | Collects all remaining elements into a vector |

**Example:**
```cpp
std::vector<Image<BGR>> images = load_dataset("path/");

auto batch = images
    | views::transform(Resize{}.width(224).height(224))
    | views::filter([](const Image<BGR>& img) { return img.cols() > 0; })
    | views::drop(10)
    | views::take(32)
    | views::to<std::vector<Image<BGR>>>();

// Or iterate lazily — one image processed at a time
for (const auto& img : images | views::transform(Resize{}.width(224).height(224))) {
    process(img);
}
```

### M3 — External Sources

| Symbol | Description |
|---|---|
| `views::VideoView{reader}` | Lazy single-pass wrapper over a `VideoReader` (non-owning; reader must outlive view) |
| `views::from_dir(path, exts)` | Lazy view over matching image files in a directory; throws `FileNotFoundError` if path missing |

Both sources compose with all M2 adapters (`transform`, `filter`, `take`, `drop`) and materialize via `views::to<std::vector<Image<BGR>>>()`.

**Examples:**
```cpp
// VideoReader source — frames processed one at a time, O(1) RAM
improc::io::VideoReader reader{"clip.mp4"};
auto frames = views::VideoView{reader}
    | views::transform(Resize{}.width(640).height(640))
    | views::take(100)
    | views::to<std::vector<Image<BGR>>>();

// Or iterate lazily (no materialization)
for (const auto& frame : views::VideoView{reader} | views::take(100)) {
    process(frame);
}

// Directory source — loads files one at a time
auto batch = views::from_dir("dataset/train/cats/", {".jpg", ".png"})
    | views::transform(Resize{}.width(224).height(224))
    | views::filter([](const Image<BGR>& img) { return !img.mat().empty(); })
    | views::to<std::vector<Image<BGR>>>();
```

---

## Planned namespaces

| Namespace | Purpose |
|---|---|
| `improc::cuda` | Wrapper over `cv::cuda` for GPU-accelerated ops |
