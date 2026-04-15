# Namespace Structure in improc++

The library is organized into modular namespaces under the root `improc` namespace. Each namespace has a single responsibility and maps to a corresponding directory under `include/improc/` and `src/`.

---

## `improc::core` — Type-safe image primitives

**Status: Implemented**

Provides a compile-time type-safe wrapper over `cv::Mat`, format conversion utilities, and a pipeline composition operator.

### Format tag types (`format_traits.hpp`)

Empty structs used as template parameters. `FormatTraits<T>` maps each tag to its OpenCV type constant and channel count.

| Tag | `cv_type` | `channels` |
|---|---|---|
| `BGR` | `CV_8UC3` | 3 |
| `Gray` | `CV_8UC1` | 1 |
| `BGRA` | `CV_8UC4` | 4 |
| `Float32` | `CV_32FC1` | 1 |
| `Float32C3` | `CV_32FC3` | 3 |

### `Image<Format>` (`image.hpp`)

Thin wrapper over `cv::Mat` with shallow-copy semantics (same as `cv::Mat`). Constructor validates that the `cv::Mat` is non-empty and its type matches `FormatTraits<Format>::cv_type` — throws `std::invalid_argument` on mismatch or empty mat.

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
```

**Geometric ops** throw `std::invalid_argument` when required parameters are missing or invalid (e.g. no dimension in `Resize`, ROI out of bounds in `Crop`, no angle in `Rotate`).

**Normalization ops** throw at construction if parameters are invalid (`NormalizeTo` requires `min < max`; `Standardize` requires `std_dev > 0`). A uniform image passed to `Normalize` or `NormalizeTo` returns a zero-filled image.

---

## `improc::io` — Input/Output

**Status: Implemented**

### `CameraCapture`

Asynchronous camera frame capture. Runs a background capture thread. Non-copyable, non-movable.

```cpp
CameraCapture cam(0);
cv::Mat frame = cam.getFrame();
cam.stop();
```

---

## `improc::ml` — Machine Learning utilities

**Status: Implemented**

### `ImageLoader`

Loads all valid images (`.jpg`, `.jpeg`, `.png`) from a directory. Returns `std::expected<std::vector<cv::Mat>, std::string>`.

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

Standalone inference functors backed by OpenCV DNN. All accept `const Image<BGR>&` and work as both standalone functors and terminal `operator|` pipeline ops. Model is loaded at construction via `cv::dnn::readNet` — throws `std::runtime_error` on failure (file not found, unsupported extension, parse error, empty net). Supported formats: `.onnx`, `.pb`, `.caffemodel`, `.weights`, `.t7`, `.net`.

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

Future: `OrtClassifier`, `OrtDetector`, `OrtForward` will provide the same API with ONNX Runtime backend — no changes to calling code needed, just swap the class name.

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

Throws `std::invalid_argument` if constructed with 0 threads. Exceptions from tasks propagate through `future.get()`.

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

`tryPop()` returns `std::optional<Result>` — `std::nullopt` if no result is ready or `start()` was not called. `start()` called twice throws `std::logic_error`. `stop()` is idempotent.

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

Values are min-max normalised to canvas height. A single value or all-equal values renders a horizontal centre line. Throws `std::invalid_argument` if vector is empty.

### `Scatter` (`visualization/scatter.hpp`)

Standalone functor. Takes `xs` and `ys` vectors and renders filled circles, each axis independently normalised.

```cpp
Image<BGR> sc = Scatter{}.title("Features").color({0, 255, 255}).point_radius(5)(xs, ys);
```

Throws `std::invalid_argument` if either vector is empty or `xs.size() != ys.size()`. An 8 px margin ensures extreme points are fully visible.

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

### Umbrella include

```cpp
#include "improc/visualization/visualization.hpp"  // includes all four
```

---

## Planned namespaces

| Namespace | Purpose |
|---|---|
| `improc::cuda` | Wrapper over `cv::cuda` for GPU-accelerated ops |
