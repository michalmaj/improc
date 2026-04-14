# Design: `improc::ml` — DNN Inference

**Date:** 2026-04-14
**Status:** Approved

## Overview

Add three inference classes to `improc::ml`: `DnnClassifier`, `DnnDetector`, and `DnnForward`. All use OpenCV DNN as the backend (`cv::dnn::Net`), are designed for future ONNX Runtime extension (new classes, no changes to existing code), and work both as standalone functors and as terminal `operator|` pipeline ops.

No new dependencies — OpenCV DNN is already available via the existing `opencv` Conan package.

## Goals

- `DnnClassifier`: image → `std::vector<ClassResult>` (top-K class id + score + label)
- `DnnDetector`: image → `std::vector<Detection>` (bounding box + class id + confidence + label), with NMS via `cv::dnn::NMSBoxes`
- `DnnForward`: image → `std::vector<float>` (raw output tensor, no interpretation)
- Fluent builder API consistent with the rest of the library
- Works as standalone functor AND terminal `operator|` pipeline op
- Error cases throw `std::runtime_error` (load failures) or `std::invalid_argument` (bad parameters)

## Non-Goals

- ONNX Runtime backend (future addition — new `OrtClassifier` etc. classes, same API shape)
- Training or fine-tuning
- Multi-input models
- Batch inference (single image per call)
- GPU inference (not ruled out, but not in scope)

---

## Shared Result Types — `include/improc/ml/result_types.hpp`

```cpp
namespace improc::ml {

struct ClassResult {
    int         class_id;  // index into labels vector (or raw index if no labels)
    float       score;     // softmax probability or raw logit, depending on model
    std::string label;     // empty string if no labels were provided
};

struct Detection {
    cv::Rect2f  box;        // pixel coordinates on the original input image
    int         class_id;
    float       confidence;
    std::string label;      // empty string if no labels were provided
};

} // namespace improc::ml
```

---

## `DnnClassifier` — `include/improc/ml/dnn_classifier.hpp`

```cpp
namespace improc::ml {

struct DnnClassifier {
    // Loads model at construction. Throws std::runtime_error on failure.
    explicit DnnClassifier(std::string model_path);

    DnnClassifier& top_k(int k);                          // default: 5; throws if k <= 0
    DnnClassifier& input_size(int w, int h);              // default: 224×224; throws if w/h <= 0
    DnnClassifier& mean(cv::Scalar m);                    // default: {0, 0, 0}
    DnnClassifier& scale(float s);                        // default: 1.0f/255.0f; throws if s <= 0
    DnnClassifier& swap_rb(bool s);                       // default: true (BGR → RGB)
    DnnClassifier& labels(std::vector<std::string> l);    // optional class names

    // Throws std::runtime_error if inference fails.
    std::vector<ClassResult> operator()(Image<BGR> img) const;

private:
    cv::dnn::Net              net_;
    int                       top_k_      = 5;
    int                       input_w_    = 224;
    int                       input_h_    = 224;
    cv::Scalar                mean_       = {0, 0, 0};
    float                     scale_      = 1.0f / 255.0f;
    bool                      swap_rb_    = true;
    std::vector<std::string>  labels_;
};

} // namespace improc::ml
```

**Rendering:** Uses `cv::dnn::blobFromImage` for preprocessing. Output blob is a `[1, num_classes]` mat; `cv::sortIdx` selects top-K indices by score.

**operator| example:**
```cpp
auto results = img | Resize{}.width(224) | DnnClassifier{"resnet50.onnx"}.top_k(3);
```

---

## `DnnDetector` — `include/improc/ml/dnn_detector.hpp`

```cpp
namespace improc::ml {

struct DnnDetector {
    enum class Style {
        YOLO,  // single output blob: [1, num_detections, 4 + num_classes]  (YOLOv5/v8/v9)
        SSD    // two output blobs: boxes [1, N, 4] + scores [1, N, num_classes]
    };

    // Loads model at construction. Throws std::runtime_error on failure.
    explicit DnnDetector(std::string model_path);

    DnnDetector& style(Style s);                          // default: Style::YOLO
    DnnDetector& output_layer(std::string name);          // YOLO: output blob name (default: auto-detect)
    DnnDetector& boxes_layer(std::string name);           // SSD: blob name for boxes
    DnnDetector& scores_layer(std::string name);          // SSD: blob name for scores
    DnnDetector& confidence_threshold(float t);           // default: 0.5f; throws if t < 0 || t > 1
    DnnDetector& nms_threshold(float t);                  // default: 0.4f; throws if t < 0 || t > 1
    DnnDetector& input_size(int w, int h);                // default: 640×640; throws if w/h <= 0
    DnnDetector& mean(cv::Scalar m);                      // default: {0, 0, 0}
    DnnDetector& scale(float s);                          // default: 1.0f/255.0f; throws if s <= 0
    DnnDetector& swap_rb(bool s);                         // default: true
    DnnDetector& labels(std::vector<std::string> l);      // optional class names

    // Throws std::runtime_error if inference fails.
    std::vector<Detection> operator()(Image<BGR> img) const;

private:
    cv::dnn::Net              net_;
    Style                     style_               = Style::YOLO;
    std::string               output_layer_;
    std::string               boxes_layer_;
    std::string               scores_layer_;
    float                     confidence_threshold_ = 0.5f;
    float                     nms_threshold_        = 0.4f;
    int                       input_w_             = 640;
    int                       input_h_             = 640;
    cv::Scalar                mean_                = {0, 0, 0};
    float                     scale_               = 1.0f / 255.0f;
    bool                      swap_rb_             = true;
    std::vector<std::string>  labels_;
};

} // namespace improc::ml
```

**NMS:** After parsing output blobs, applies `cv::dnn::NMSBoxes` with `confidence_threshold_` and `nms_threshold_`. Box coordinates are scaled back to original image dimensions.

**YOLO output parsing:** Output blob shape `[1, num_det, 4+num_classes]`; first 4 values are `cx, cy, w, h` (normalized). Class confidence = max over class scores.

**SSD output parsing:** Boxes blob `[1, N, 4]` as `[y1, x1, y2, x2]` normalized; scores blob `[1, N, C]`; argmax over C for class id.

**Exotic formats:** Use `DnnForward` and parse manually.

---

## `DnnForward` — `include/improc/ml/dnn_forward.hpp`

```cpp
namespace improc::ml {

struct DnnForward {
    // Loads model at construction. Throws std::runtime_error on failure.
    explicit DnnForward(std::string model_path);

    DnnForward& input_size(int w, int h);   // default: 224×224; throws if w/h <= 0
    DnnForward& mean(cv::Scalar m);         // default: {0, 0, 0}
    DnnForward& scale(float s);             // default: 1.0f/255.0f; throws if s <= 0
    DnnForward& swap_rb(bool s);            // default: true

    // Runs forward pass. Returns flattened output tensor.
    // Throws std::runtime_error if inference fails.
    std::vector<float> operator()(Image<BGR> img) const;

private:
    cv::dnn::Net  net_;
    int           input_w_ = 224;
    int           input_h_ = 224;
    cv::Scalar    mean_     = {0, 0, 0};
    float         scale_    = 1.0f / 255.0f;
    bool          swap_rb_  = true;
};

} // namespace improc::ml
```

---

## Error Handling Summary

| Situation | Behavior |
|---|---|
| Non-existent or unreadable model file | `std::runtime_error` in constructor |
| Unsupported model format | `std::runtime_error` in constructor |
| `top_k(0)` or negative | `std::invalid_argument` |
| `scale(0)` or negative | `std::invalid_argument` |
| `confidence_threshold` outside [0, 1] | `std::invalid_argument` |
| `nms_threshold` outside [0, 1] | `std::invalid_argument` |
| `input_size(0, *)` or `(*, 0)` | `std::invalid_argument` |
| Inference fails (shape mismatch, etc.) | `std::runtime_error` in `operator()` |

---

## Tests (error path only — no test model required)

### `test_dnn_classifier.cpp`
- Non-existent path throws `std::runtime_error`
- Invalid extension throws `std::runtime_error`
- `top_k(0)` throws `std::invalid_argument`
- `top_k(-1)` throws `std::invalid_argument`
- `scale(0)` throws `std::invalid_argument`
- `scale(-1)` throws `std::invalid_argument`
- `input_size(0, 224)` throws `std::invalid_argument`
- `input_size(224, 0)` throws `std::invalid_argument`

### `test_dnn_detector.cpp`
- Non-existent path throws `std::runtime_error`
- `confidence_threshold(-0.1f)` throws `std::invalid_argument`
- `confidence_threshold(1.1f)` throws `std::invalid_argument`
- `nms_threshold(-0.1f)` throws `std::invalid_argument`
- `nms_threshold(1.1f)` throws `std::invalid_argument`
- `scale(0)` throws `std::invalid_argument`
- `input_size(0, 640)` throws `std::invalid_argument`

### `test_dnn_forward.cpp`
- Non-existent path throws `std::runtime_error`
- `scale(0)` throws `std::invalid_argument`
- `input_size(0, 224)` throws `std::invalid_argument`
- `input_size(224, 0)` throws `std::invalid_argument`

---

## File Structure

```
include/improc/ml/
    result_types.hpp
    dnn_classifier.hpp
    dnn_detector.hpp
    dnn_forward.hpp

src/ml/
    dnn_classifier.cpp
    dnn_detector.cpp
    dnn_forward.cpp

tests/ml/
    test_dnn_classifier.cpp
    test_dnn_detector.cpp
    test_dnn_forward.cpp

examples/ml/
    demo_dnn_inference.cpp
```

`CMakeLists.txt`: add `demo_dnn_inference` executable. Sources and tests are auto-discovered via `GLOB_RECURSE CONFIGURE_DEPENDS`.

---

## Future Extension (ONNX Runtime)

When `onnxruntime` becomes available, add:
- `OrtClassifier` / `OrtDetector` / `OrtForward` — same API shape, ORT backend
- No changes to existing `Dnn*` classes or `result_types.hpp`
- User switches backend by swapping class name only
