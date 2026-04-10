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

### `Image<Format>` (`image.hpp`)

Thin wrapper over `cv::Mat` with shallow-copy semantics (same as `cv::Mat`). Constructor validates that the `cv::Mat` type matches `FormatTraits<Format>::cv_type` — throws `std::invalid_argument` on mismatch.

```cpp
Image<BGR> img(mat);          // validated at construction
Image<BGR> copy = img;        // shallow copy (shared data)
Image<BGR> owned = img.clone(); // deep copy
cv::Mat& raw = img.mat();     // explicit interop with OpenCV
```

### `convert<To, From>()` (`convert.hpp`)

Free function template. Primary template is `= delete` — unsupported conversions are compile errors. Explicit specializations define allowed conversions:

- `BGR` ↔ `Gray`
- `BGR` ↔ `BGRA`
- `Gray` → `Float32` (normalized 0–1)

```cpp
Image<Gray> gray = convert<Gray, BGR>(bgr_image);
```

### Pipeline `operator|` (`pipeline.hpp`)

Composes processing steps using the pipe operator. Each step is a small functor.

```cpp
Image<Float32> result = bgr_image | ToGray{} | ToFloat32{};
```

Available functors: `ToGray`, `ToBGR`, `ToFloat32`.

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
| `improc::pipeline` | *(merge into `improc::core`)* Additional pipeline ops: Resize, Rotate, Crop |
| `improc::threading` | ThreadPool, producer-consumer frame pipeline |
| `improc::cuda` | Wrapper over `cv::cuda` for GPU-accelerated ops |
