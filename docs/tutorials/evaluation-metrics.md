# Evaluation Metrics

This tutorial shows how to measure model quality using improc++ evaluation accumulators: `ClassEval` for classifiers, `DetectionEval` for object detectors, and `TrackingEval` for multi-object trackers.

## Prerequisites

- Completed [ML Inference](ml-inference.md)
- A validation set with ground-truth labels

## Classification — `ClassEval`

`ClassEval` accumulates per-frame predictions and ground-truth class IDs, then computes accuracy, per-class precision/recall/F1, and the confusion matrix.

```cpp
#include "improc/ml/ml.hpp"
#include "improc/io/image_io.hpp"

using namespace improc::ml;
using namespace improc::io;

int main() {
    std::vector<std::string> labels = {"cat", "dog", "bird"};

    ClassEval eval{};
    eval.class_names(labels);

    DnnClassifier cls{"resnet50.onnx"};
    cls.input_size(224, 224).scale(1.0f / 255.0f).swap_rb(true).labels(labels);

    // Simulate a validation loop
    for (const auto& [path, gt_id] : val_set) {
        auto img = imread<BGR>(path);
        if (!img) continue;
        auto results = cls(*img);   // std::vector<ClassResult>
        if (results.empty()) continue;
        eval.update(results[0].class_id, gt_id);
    }

    auto metrics = eval.compute();
    std::cout << "Accuracy: " << metrics.accuracy << "\n";

    for (const auto& [cls_name, f1] : metrics.per_class_f1)
        std::cout << cls_name << "  F1=" << f1 << "\n";
}
```

`ClassMetrics` fields:
| Field | Type | Description |
|---|---|---|
| `accuracy` | `float` | Overall fraction correct |
| `per_class_precision` | `map<string, float>` | Precision per class |
| `per_class_recall` | `map<string, float>` | Recall per class |
| `per_class_f1` | `map<string, float>` | F1 per class |
| `confusion_matrix` | `ConfusionMatrix` | `cv::Mat_<int>` with class counts |

### Reset and Re-use

```cpp
eval.reset();  // clears all accumulated state; class_names is preserved
```

## One-off Metric Functions

For quick calculations without an accumulator:

```cpp
#include "improc/ml/ml.hpp"
using namespace improc::ml;

std::vector<int> preds = {0, 1, 2, 0, 1};
std::vector<int>  gts  = {0, 1, 1, 0, 2};

float acc  = accuracy(preds, gts);                   // 0.6
float prec = precision_score(preds, gts, /*class=*/1); // precision for class 1
float rec  = recall_score(preds, gts, 1);
float f1   = f1_score(preds, gts, 1);
```

## Object Detection — `DetectionEval`

`DetectionEval` computes COCO-style mAP@50 and mAP@[50:95] across multiple IoU thresholds.

```cpp
#include "improc/ml/ml.hpp"
using namespace improc::ml;

DetectionEval eval{};

DnnDetector det{"yolov8n.onnx"};
det.input_size(640, 640).confidence_threshold(0.25f).nms_threshold(0.4f).labels(labels);

for (const auto& [img_path, gt_boxes] : val_dataset) {
    auto img = imread<BGR>(img_path);
    if (!img) continue;

    std::vector<Detection> preds = det(*img);
    eval.update(preds, gt_boxes);
}

auto metrics = eval.compute();
std::cout << "mAP@50:    " << metrics.mAP_50    << "\n";
std::cout << "mAP@50:95: " << metrics.mAP_50_95 << "\n";

for (const auto& [cls, ap] : metrics.per_class_AP)
    std::cout << cls << "  AP=" << ap << "\n";
```

`DetectionMetrics` fields:
| Field | Type | Description |
|---|---|---|
| `mAP_50` | `float` | Mean AP at IoU threshold 0.50 |
| `mAP_50_95` | `float` | Mean AP averaged over IoU 0.50–0.95 |
| `per_class_AP` | `map<string, float>` | AP@50 per class |

### Ground-Truth Format

Ground-truth boxes are `std::vector<BBox>`, where `BBox` holds a class label alongside the bounding box:

```cpp
BBox gt;
gt.box       = cv::Rect2f{x, y, w, h};   // pixel coordinates
gt.class_id  = 0;
gt.label     = "person";
```

### Precision–Recall Curves

`DetectionEval::pr_curves()` returns recall and precision vectors per class at IoU=0.50, useful for plotting:

```cpp
auto curves = eval.pr_curves();
// curves["person"] = {recalls[], precisions[]}
```

### IoU Helper

```cpp
BBox a, b;  // two detections
float overlap = iou(a, b);  // returns 0.0 if either box has zero area
```

## Tracking — `TrackingEval`

`TrackingEval` accumulates CLEAR MOT metrics (MOTA, MOTP) and IDF1 across frames.

```cpp
#include "improc/ml/tracking/tracking_eval.hpp"
#include "improc/ml/tracking/tracking.hpp"
using namespace improc::ml;

TrackingEval eval{};
eval.iou_threshold(0.5f);   // default: 0.5

ByteTracker tracker{};

// Per-frame loop
for (auto& frame : video) {
    std::vector<Detection> dets = detector(frame.image);
    std::vector<Track>     tracks = tracker.update(dets);

    // Ground truth for this frame: vector<TrackGT> with persistent instance IDs
    eval.update(tracks, frame.ground_truth);
}

auto metrics = eval.compute();
std::cout << "MOTA:      " << metrics.MOTA      << "\n";
std::cout << "MOTP:      " << metrics.MOTP      << "\n";
std::cout << "IDF1:      " << metrics.IDF1      << "\n";
std::cout << "Precision: " << metrics.Precision << "\n";
std::cout << "Recall:    " << metrics.Recall    << "\n";
std::cout << "FP:  " << metrics.FP   << "\n";
std::cout << "FN:  " << metrics.FN   << "\n";
std::cout << "IDSW:" << metrics.IDSW << "\n";
```

`TrackingMetrics` fields:
| Field | Type | Description |
|---|---|---|
| `MOTA` | `float` | 1 − (FN + FP + IDSW) / GT_total |
| `MOTP` | `float` | Mean IoU of matched pairs |
| `IDF1` | `float` | 2·IDTP / (2·IDTP + IDFP + IDFN) |
| `Precision` | `float` | TP / (TP + FP) |
| `Recall` | `float` | TP / (TP + FN) |
| `FP` / `FN` / `IDSW` | `int` | Raw false positives / negatives / ID switches |

### Ground-Truth Format

```cpp
TrackGT gt;
gt.id   = 42;                               // persistent instance ID across frames
gt.bbox = BBox{.box = cv::Rect2f{...}};
```

## Combining Evaluators

All three evaluators have the same lifecycle: construct → loop calling `update()` → call `compute()` → optionally `reset()`.

```cpp
ClassEval     cls_eval{};
DetectionEval det_eval{};
TrackingEval  trk_eval{};

// ... run through dataset ...

auto cls_m = cls_eval.compute();
auto det_m = det_eval.compute();
auto trk_m = trk_eval.compute();
```

## Next Steps

- [ML Visualization Charts](ml-visualization-charts.md) — plot confusion matrices, PR/ROC curves, and AP bars
- [Multi-Object Tracking](multi-object-tracking.md) — run trackers on video
- [ML Inference](ml-inference.md) — run models that feed these evaluators
