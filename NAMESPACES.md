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
- `Gray` → `Float32` (scale 1/255)
- `BGR` → `Float32C3` (scale 1/255, 3-channel float)

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

---

## Planned namespaces

| Namespace | Purpose |
|---|---|
| `improc::threading` | ThreadPool, producer-consumer frame pipeline |
| `improc::visualization` | Histogram and chart plotting utilities |
| `improc::cuda` | Wrapper over `cv::cuda` for GPU-accelerated ops |
