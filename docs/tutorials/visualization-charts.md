# Visualization Charts

This tutorial covers the basic visualization tools in `improc::visualization`: pixel intensity histograms, line plots, scatter charts, image montages, and bounding-box annotation.

## Prerequisites

- Completed [Getting Started](getting-started.md)
- `improc/visualization/visualization.hpp` (umbrella include for all chart types)

## Histogram

`Histogram` renders a pixel-intensity distribution chart. It accepts `Image<BGR>` (three overlapping channel curves), `Image<Gray>` (single curve), or `Image<Float32>` (single curve, values in [0,1]).

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"
#include "improc/visualization/visualization.hpp"

using namespace improc::core;
using namespace improc::io;
using namespace improc::visualization;

int main() {
    auto src = imread<BGR>("photo.png");
    if (!src) { std::cerr << src.error().message << "\n"; return 1; }

    // BGR histogram — three channel curves, pipeline form
    Image<BGR> chart = *src | Histogram{}.bins(256).width(512).height(256);
    chart | Show{"BGR Histogram"};

    // Grayscale histogram
    auto gray = convert<Gray>(*src);
    Image<BGR> gray_chart = gray | Histogram{}.bins(128).width(400).height(200);
    gray_chart | Show{"Gray Histogram"};

    cv::waitKey(0);
}
```

Defaults: `bins=256`, `width=512`, `height=256`. The histogram is always returned as `Image<BGR>`.

## LinePlot

`LinePlot` draws a 2D line chart from a `std::vector<float>`. The X-axis spans indices 0…N-1; the Y-axis auto-scales to the data range.

```cpp
#include "improc/visualization/visualization.hpp"
using namespace improc::visualization;

// Training loss curve
std::vector<float> losses = {1.45f, 1.12f, 0.87f, 0.73f, 0.61f, 0.52f, 0.44f};

Image<BGR> chart = LinePlot{}
    .title("Training Loss")
    .color({50, 180, 250})   // BGR — amber
    .width(640)
    .height(360)
    (losses);

chart | Show{"Loss Curve"};

// Save to file
imwrite("loss_curve.png", chart);
```

Throws `improc::ParameterError` if the input vector is empty. Defaults: `width=640`, `height=360`, white line, no title.

## Scatter

`Scatter` plots `(x, y)` point pairs. Pass two parallel `std::vector<float>` arrays of equal length.

```cpp
#include "improc/visualization/visualization.hpp"
using namespace improc::visualization;

// Predicted vs. ground-truth scores
std::vector<float> xs = {0.2f, 0.5f, 0.8f, 0.3f, 0.9f, 0.6f};
std::vector<float> ys = {0.18f, 0.52f, 0.76f, 0.35f, 0.88f, 0.65f};

Image<BGR> chart = Scatter{}
    .title("Predicted vs. Ground Truth")
    .color({0, 200, 255})    // BGR — yellow
    .point_radius(4)
    .width(512)
    .height(512)
    (xs, ys);

chart | Show{"Scatter"};
```

Throws `improc::ParameterError` if either vector is empty or their sizes differ. Defaults: `width=512`, `height=512`, cyan points, `point_radius=3`.

## Montage

`Montage` arranges a collection of `Image<BGR>` into a grid — useful for inspecting a batch of images or augmentation results side-by-side.

```cpp
#include "improc/visualization/visualization.hpp"
using namespace improc::visualization;
using namespace improc::views;

// Load a directory of images
std::vector<Image<BGR>> batch = from_dir("data/cats/", {".png", ".jpg"})
    | transform(Resize{}.width(128).height(128))
    | to<std::vector<Image<BGR>>>();

Image<BGR> grid = Montage{batch}
    .cols(4)               // 4 columns — rows computed automatically
    .cell_size(128, 128)   // resize each cell (default: size of first image)
    .gap(4)                // pixel gap between cells
    .background({30, 30, 30})  // BGR dark grey background
    ();                    // render

grid | Show{"Batch"};
imwrite("montage.png", grid);
```

Defaults: `cols = ceil(sqrt(n))`, `cell_size` = first image dimensions, `gap = 0`, black background. Throws `ParameterError` if the image vector is empty.

## DrawBoundingBoxes

`DrawBoundingBoxes` annotates detection results onto a BGR image. It draws boxes, labels, and confidence scores without modifying the source.

```cpp
#include "improc/ml/ml.hpp"
#include "improc/visualization/visualization.hpp"
using namespace improc::ml;
using namespace improc::visualization;

DnnDetector det{"yolov8n.onnx"};
det.input_size(640, 640).confidence_threshold(0.4f).labels({"person", "car"});

auto frame = imread<BGR>("street.png");
std::vector<Detection> dets = det(*frame);

// Draw — pipeline form
Image<BGR> annotated = *frame | DrawBoundingBoxes{dets}
    .color({0, 255, 0})        // BGR green (default)
    .thickness(2)
    .font_scale(0.5)
    .show_label(true)
    .show_confidence(true);

annotated | Show{"Detections"};
```

The source image is never modified — `DrawBoundingBoxes` returns a new clone with the annotations.

## Saving Charts

All chart functors return `Image<BGR>`, which can be saved directly:

```cpp
imwrite("histogram.png", chart);
imwrite("montage.png",   grid);
```

## Next Steps

- [ML Visualization Charts](ml-visualization-charts.md) — confusion matrices, PR/ROC curves, IoU histograms
- [Building a Pipeline](building-a-pipeline.md) — format conversions, views, core ops
- [ML Inference](ml-inference.md) — the detector that produces `Detection` results
