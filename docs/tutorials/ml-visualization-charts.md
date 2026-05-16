# ML Visualization Charts

This tutorial shows how to plot model evaluation results using the five ML chart functors introduced in v0.5.0: `ConfusionMatrixPlot`, `PRCurvePlot`, `ROCCurvePlot`, `ClassBarChart`, and `IoUHistogram`.

## Prerequisites

- Completed [Evaluation Metrics](evaluation-metrics.md)
- `improc/visualization/ml_charts.hpp` or individual chart headers

## Overview

All five charts follow the same pattern:

1. Construct with data (computed metrics or raw vectors).
2. Optionally configure size and title via fluent setters.
3. Call `operator()()` to get an `Image<BGR>`.
4. Display or save via the pipeline.

```cpp
#include "improc/visualization/ml_charts.hpp"
#include "improc/visualization/visualization.hpp"   // Show
using namespace improc::visualization;
using namespace improc::ml;
```

## Confusion Matrix — `ConfusionMatrixPlot`

`ConfusionMatrixPlot` renders a grid with colour-coded counts. Diagonal cells are tinted toward green; off-diagonal cells graduate from amber (small errors) to red (severe).

### From `ClassMetrics`

```cpp
ClassEval eval{};
eval.class_names({"cat", "dog", "bird"});
// ... eval.update() calls ...
auto metrics = eval.compute();

ConfusionMatrixPlot plot{metrics.confusion_matrix};
plot.width(480).height(480).title("Validation Set");

auto chart = plot();       // Image<BGR>
chart | Show{"Confusion Matrix"};
```

### From `ClassMetrics` directly

```cpp
ConfusionMatrixPlot plot{metrics};   // uses confusion_matrix + per_class names
```

### From raw counts

```cpp
std::vector<std::vector<int>> counts = {
    {50,  3,  2},
    { 4, 44,  7},
    { 1,  5, 49},
};
std::vector<std::string> labels = {"cat", "dog", "bird"};

ConfusionMatrixPlot plot{counts, labels};
auto chart = plot();
```

## Precision–Recall Curve — `PRCurvePlot`

`PRCurvePlot` draws one curve per class with AP (area under PR curve, COCO 101-point interpolation) shown in the legend.

### From `DetectionEval`

```cpp
DetectionEval det_eval{};
// ... det_eval.update() calls ...
auto curves = det_eval.pr_curves();   // map<string, pair<recalls, precisions>>

PRCurvePlot plot{curves};
plot.width(640).height(480).title("PR Curve @ IoU=0.50");

auto chart = plot();
chart | Show{"PR Curve"};
```

### From separate maps

```cpp
std::map<std::string, std::vector<float>> recalls, precisions;
recalls["person"]    = {0.0f, 0.5f, 0.8f, 1.0f};
precisions["person"] = {1.0f, 0.9f, 0.7f, 0.5f};

PRCurvePlot plot{recalls, precisions};
```

### Optional mAP annotation

```cpp
auto det_metrics = det_eval.compute();
PRCurvePlot plot{curves};
plot.mAP_50(det_metrics.mAP_50);   // adds mAP@50 annotation to chart title
```

## ROC Curve — `ROCCurvePlot`

`ROCCurvePlot` displays TPR vs FPR curves for multi-class problems, with AUC per class in the legend (computed via the trapezoid rule).

`ClassEval` stores only confusion-matrix counts, not raw per-threshold scores. You supply FPR/TPR vectors from your own threshold sweep:

```cpp
// Compute FPR/TPR sweep externally, then hand to the plot:
std::map<std::string, std::pair<std::vector<float>, std::vector<float>>> roc_curves;
roc_curves["cat"] = { {0.0f, 0.1f, 0.5f, 1.0f},   // fpr
                      {0.0f, 0.8f, 0.9f, 1.0f} };  // tpr
roc_curves["dog"] = { {0.0f, 0.2f, 0.6f, 1.0f},
                      {0.0f, 0.7f, 0.85f, 1.0f} };

ROCCurvePlot plot{roc_curves};
plot.width(640).height(480).title("ROC — Validation");

auto chart = plot();
chart | Show{"ROC Curve"};
```

Alternatively, build from separate FPR and TPR maps:

```cpp
std::map<std::string, std::vector<float>> fprs, tprs;
fprs["cat"] = {0.0f, 0.1f, 0.5f, 1.0f};
tprs["cat"] = {0.0f, 0.8f, 0.9f, 1.0f};

ROCCurvePlot plot{fprs, tprs};
```

## Class Bar Chart — `ClassBarChart`

`ClassBarChart` renders grouped bars showing Precision, Recall, and F1 (or AP) per class.

### From `ClassMetrics`

```cpp
ClassEval cls_eval{};
cls_eval.class_names({"cat", "dog", "bird"});
// ... updates ...
auto cls_metrics = cls_eval.compute();

ClassBarChart plot{cls_metrics};
plot.width(640).height(400).title("Per-Class P/R/F1");

auto chart = plot();
chart | Show{"Class Bar Chart"};
```

### From `DetectionMetrics` (single AP bar per class)

```cpp
DetectionEval det_eval{};
// ... updates ...
auto det_metrics = det_eval.compute();

ClassBarChart plot{det_metrics};   // renders AP@50 bar per class
plot.title("Per-Class AP@50");
auto chart = plot();
```

### From raw data

```cpp
// map<class_name, array<P, R, F1>>
std::map<std::string, std::array<float, 3>> data = {
    {"cat",  {0.92f, 0.88f, 0.90f}},
    {"dog",  {0.85f, 0.90f, 0.87f}},
    {"bird", {0.78f, 0.82f, 0.80f}},
};

ClassBarChart plot{data};
auto chart = plot();
```

## IoU Histogram — `IoUHistogram`

`IoUHistogram` plots the distribution of IoU scores between predictions and matched ground-truth boxes. Bars above the threshold are violet; bars below are amber.

```cpp
// Collect IoU scores from a validation run
std::vector<float> ious;
for (const auto& [pred, gt] : matched_pairs)
    ious.push_back(iou(pred, gt));

IoUHistogram plot{ious};
plot.width(800)
    .height(300)
    .bins(20)
    .threshold(0.5f)
    .title("IoU Distribution — Validation");

auto chart = plot();
chart | Show{"IoU Histogram"};
```

Throws `improc::ParameterError` if `ious` is empty, `bins < 1`, `threshold` outside [0,1], or dimensions are non-positive.

## Saving Charts to File

All chart operators return `Image<BGR>`, which can be saved with `imwrite`:

```cpp
#include "improc/io/image_io.hpp"
using namespace improc::io;

auto chart = ConfusionMatrixPlot{metrics}.width(480).height(480)();
imwrite("confusion_matrix.png", chart);
```

## Full Evaluation Report Example

```cpp
#include "improc/ml/ml.hpp"
#include "improc/visualization/ml_charts.hpp"
#include "improc/visualization/visualization.hpp"
#include "improc/io/image_io.hpp"
using namespace improc::ml;
using namespace improc::visualization;
using namespace improc::io;

int main() {
    // --- Run evaluation ---
    ClassEval     cls_eval{};
    DetectionEval det_eval{};
    cls_eval.class_names({"cat", "dog", "bird"});

    DnnClassifier cls{"resnet50.onnx"};
    DnnDetector   det{"yolov8n.onnx"};
    cls.input_size(224, 224).scale(1.0f/255.0f).swap_rb(true);
    det.input_size(640, 640).confidence_threshold(0.25f);

    for (const auto& sample : val_set) {
        auto img = imread<BGR>(sample.path);
        if (!img) continue;
        auto cls_res = cls(*img);
        if (!cls_res.empty()) cls_eval.update(cls_res[0].class_id, sample.cls_gt);
        det_eval.update(det(*img), sample.det_gt);
    }

    // --- Compute metrics ---
    auto cls_m = cls_eval.compute();
    auto det_m = det_eval.compute();
    auto pr    = det_eval.pr_curves();

    // --- Render charts ---
    ConfusionMatrixPlot{cls_m}.width(480).height(480).title("Confusion Matrix")()
        | Show{"Confusion Matrix"};

    ClassBarChart{cls_m}.width(640).height(400).title("P/R/F1 per class")()
        | Show{"Class Metrics"};

    PRCurvePlot{pr}.mAP_50(det_m.mAP_50).title("PR Curve")()
        | Show{"PR Curve"};

    ClassBarChart{det_m}.title("AP@50 per class")()
        | Show{"Detection AP"};

    cv::waitKey(0);
}
```

## Next Steps

- [Multi-Object Tracking](multi-object-tracking.md) — visualize tracks with `DrawTracks`
- [Evaluation Metrics](evaluation-metrics.md) — how the underlying numbers are computed
- [ML Inference](ml-inference.md) — the models that produce these predictions
