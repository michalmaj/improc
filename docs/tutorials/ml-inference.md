# ML Inference

This tutorial shows how to run classification and object detection models using improc++. Both the OpenCV DNN backend (`DnnClassifier`, `DnnDetector`) and the ONNX Runtime backend (`OnnxClassifier`, `OnnxDetector`) are covered.

## Prerequisites

- Completed [Getting Started](getting-started.md)
- A trained model file — `.onnx`, `.pb`, `.caffemodel`, or `.weights`
- Optional: class label file (one label per line)

## Choosing a Backend

| Backend | Header | When to use |
|---|---|---|
| OpenCV DNN | `improc/ml/ml.hpp` | Caffe, TensorFlow, Darknet, ONNX (no EP) |
| ONNX Runtime | `improc/onnx/onnx.hpp` | ONNX models; CoreML EP on Apple Silicon |

Both backends expose the same fluent API and return the same `ClassResult` / `Detection` types — you can swap them with a one-line change.

## Classification

### DnnClassifier (OpenCV DNN)

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"
#include "improc/ml/ml.hpp"

using namespace improc::core;
using namespace improc::ml;
using namespace improc::io;

int main() {
    // Load class labels (one per line, optional)
    std::vector<std::string> labels = {"cat", "dog", "bird"};

    // Create classifier — throws ModelError on load failure
    DnnClassifier cls{"resnet50.onnx"};
    cls.input_size(224, 224)
       .scale(1.0f / 255.0f)
       .swap_rb(true)           // BGR → RGB
       .labels(labels)
       .top_k(3);

    // Load image and classify — pipeline-composable
    auto src = imread<BGR>("photo.jpg");
    if (!src) { std::cerr << src.error().message << "\n"; return 1; }

    auto results = *src | Resize{}.width(224).height(224) | cls;
    // results is std::vector<ClassResult>

    for (const auto& r : results)
        std::cout << r.label << "  score=" << r.score << "\n";
}
```

`ClassResult` fields: `class_id` (int), `score` (float, higher is better), `label` (string, empty if no labels set).

### OnnxClassifier (ONNX Runtime)

```cpp
#include "improc/onnx/onnx.hpp"
using namespace improc::onnx;

OnnxClassifier cls{"mobilenetv3.onnx"};
cls.input_size(224, 224)
   .mean(0.485f, 0.456f, 0.406f)  // ImageNet mean, BGR channel order
   .scale(1.0f / 255.0f)
   .swap_rb(true)
   .labels(labels)
   .top_k(5);

auto result = cls(img);  // returns std::expected<std::vector<ClassResult>, Error>
if (!result) { std::cerr << result.error().message << "\n"; return 1; }

for (const auto& r : *result)
    std::cout << r.label << "  " << r.score << "\n";
```

On Apple Silicon, CoreML is used automatically for supported ops with CPU fallback.

## Object Detection

### DnnDetector (OpenCV DNN)

```cpp
#include "improc/ml/ml.hpp"
#include "improc/visualization/visualization.hpp"

using namespace improc::ml;
using namespace improc::core;
using namespace improc::visualization;

int main() {
    std::vector<std::string> labels = {"person", "car", "bicycle"};

    // YOLO-style model (default) — single output blob [1, N, 5+C]
    DnnDetector det{"yolov8n.onnx"};
    det.input_size(640, 640)
       .confidence_threshold(0.5f)
       .nms_threshold(0.4f)
       .labels(labels);

    auto src = imread<BGR>("street.jpg");
    if (!src) return 1;

    std::vector<Detection> dets = det(*src);
    // dets: vector<Detection> — each has .box, .class_id, .confidence, .label

    // Draw and display
    *src | DrawBoundingBoxes{dets}.thickness(2) | Show{"Detections"};
}
```

`Detection` fields: `box` (cv::Rect2f, pixel coordinates), `class_id` (int), `confidence` (float), `label` (string).

### SSD-style model

```cpp
DnnDetector ssd{"ssd_mobilenet.pb"};
ssd.style(DnnDetector::Style::SSD)
   .boxes_layer("detection_boxes")
   .scores_layer("detection_scores")
   .confidence_threshold(0.5f)
   .labels(labels);

auto dets = ssd(frame);
```

### OnnxDetector (ONNX Runtime)

```cpp
#include "improc/onnx/onnx.hpp"
using namespace improc::onnx;

OnnxDetector det{"yolov8n.onnx"};
det.input_size(640, 640)
   .confidence_threshold(0.5f)
   .nms_threshold(0.4f)
   .labels(labels);

auto result = det(frame);   // std::expected<std::vector<Detection>, Error>
if (!result) return 1;

frame | DrawBoundingBoxes{*result} | Show{"OnnxDetections"};
```

YOLOv8 ONNX models (exported via `model.export(format="onnx")`) are detected automatically by shape heuristic — no extra configuration needed.

## Face Detection with Haar Cascades

For classic face detection, use `HaarCascadeLoader`:

```cpp
#include "improc/ml/ml.hpp"

HaarCascadeLoader loader{"haarcascade_frontalface_default.xml"};
auto classifier = loader.get();  // std::expected<cv::CascadeClassifier, Error>
if (!classifier) return 1;

cv::Mat gray;
cv::cvtColor(frame.mat(), gray, cv::COLOR_BGR2GRAY);
std::vector<cv::Rect> faces;
classifier->detectMultiScale(gray, faces);
```

Valid extensions: `.xml`, `.yml`, `.yaml`.

## Raw Tensor Output

For models with custom output formats, use `DnnForward` to get the raw output blob:

```cpp
DnnForward fwd{"encoder.onnx"};
fwd.input_size(128, 128).scale(1.0f / 255.0f);

std::vector<float> embedding = fwd(img);  // flat float vector
```

## Error Handling

Model load failures throw `improc::ModelError`:

```cpp
try {
    DnnClassifier cls{"missing.onnx"};
} catch (const improc::ModelError& e) {
    std::cerr << e.what() << "\n";  // path + reason
}
```

`OnnxClassifier` / `OnnxDetector` return `std::expected` — check `!result` before use.

## Next Steps

- [Augmentation Pipeline](augmentation-pipeline.md) — prepare training data
- [Evaluation Metrics](evaluation-metrics.md) — measure your model's accuracy
- [Multi-Object Tracking](multi-object-tracking.md) — track detections across frames
