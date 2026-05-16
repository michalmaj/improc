# Image Analysis

This tutorial covers classical image analysis tools in `improc::core`: contour extraction, connected-component labelling, distance transform, and morphological shape analysis.

## Prerequisites

- Completed [Building a Pipeline](building-a-pipeline.md)
- `improc/core/pipeline.hpp` (umbrella include)

## Contour Analysis

Contours are extracted from a binary (thresholded) `Image<Gray>`. `FindContours` returns a `ContourSet` — a collection of point-chains, one per detected boundary.

### Extract Contours

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"
using namespace improc::core;
using namespace improc::io;

int main() {
    auto src = imread<BGR>("shapes.png");
    if (!src) return 1;

    // Threshold to binary — FindContours requires Image<Gray>
    Image<Gray> binary = convert<Gray>(*src)
        | Threshold{}.value(128.0).mode(ThresholdMode::Binary);

    // External: only outer contours (default)
    ContourSet cs = binary | FindContours{};

    // Tree: full contour hierarchy (outer + holes)
    ContourSet tree = binary | FindContours{}
        .mode(FindContours::Mode::Tree)
        .method(FindContours::Method::Simple);

    std::cout << "Found " << cs.size() << " contours\n";
    for (std::size_t i = 0; i < cs.size(); ++i) {
        std::cout << "  [" << i << "]  area="     << cs.area(i)
                  << "  perimeter=" << cs.perimeter(i)
                  << "  bbox="      << cs.bounding_rect(i) << "\n";
    }
}
```

`FindContours::Mode`:
- `External` — outermost contours only (default, fastest)
- `List` — all contours, no hierarchy
- `CComp` — two-level hierarchy (outer + holes)
- `Tree` — full hierarchy tree

### Filter by Area

```cpp
ContourSet cs = binary | FindContours{};

// Keep only large contours
std::vector<std::size_t> large;
for (std::size_t i = 0; i < cs.size(); ++i)
    if (cs.area(i) > 500.0) large.push_back(i);

std::cout << large.size() << " large contours (area > 500 px)\n";
```

### Draw Contours

```cpp
#include "improc/visualization/visualization.hpp"
using namespace improc::visualization;

// Draw all contours in green (default)
Image<BGR> annotated = *src | DrawContours{cs};

// Draw only the largest contour, thickness 2
Image<BGR> one = *src | DrawContours{cs}
    .index(0)
    .color({0, 0, 255})  // red
    .thickness(2);

// Fill all contours
Image<BGR> filled = *src | DrawContours{cs}
    .color({0, 255, 0})
    .thickness(-1);      // -1 = fill

annotated | Show{"Contours"};
```

## Connected Components

`ConnectedComponents` labels each connected region in a binary image with a unique integer ID. Label `0` is always the background.

```cpp
#include "improc/core/pipeline.hpp"
using namespace improc::core;

Image<Gray> binary = convert<Gray>(*src) | Threshold{}.mode(ThresholdMode::Otsu);

ComponentMap cm = binary | ConnectedComponents{};

// 4-connectivity (default: 8)
ComponentMap cm4 = binary | ConnectedComponents{}
    .connectivity(ConnectedComponents::Connectivity::Four);

std::cout << "Components (incl. background): " << cm.count() << "\n";

// Skip label 0 (background)
for (int i = 1; i < cm.count(); ++i) {
    std::cout << "Component " << i
              << "  area="     << cm.area(i)
              << "  centroid=" << cm.centroid(i)
              << "  bbox="     << cm.bounding_rect(i) << "\n";
}
```

### Extract a Component Mask

`cm.mask(i)` returns a CV_8U binary mat (255 where `labels == i`, 0 elsewhere):

```cpp
// Isolate the largest non-background component
int largest = 1;
for (int i = 2; i < cm.count(); ++i)
    if (cm.area(i) > cm.area(largest)) largest = i;

cv::Mat mask = cm.mask(largest);   // raw cv::Mat, CV_8U
Image<Gray> isolated_mask{mask};
```

### Visualise All Components

```cpp
// Color-code each component for inspection
cv::Mat colored(binary.mat().size(), CV_8UC3, cv::Scalar{0, 0, 0});
std::vector<cv::Scalar> colors = {
    {255, 0, 0}, {0, 255, 0}, {0, 0, 255},
    {255, 255, 0}, {0, 255, 255}, {255, 0, 255}
};
for (int i = 1; i < cm.count(); ++i) {
    cv::Scalar col = colors[(i - 1) % colors.size()];
    colored.setTo(col, cm.mask(i));
}
Image<BGR>{colored} | Show{"Components"};
```

## Distance Transform

`DistanceTransform` computes, for each non-zero pixel, its distance to the nearest background (zero) pixel. Returns `Image<Float32>`.

```cpp
#include "improc/core/pipeline.hpp"
using namespace improc::core;

Image<Gray> binary = /* binary mask, e.g. from Threshold */;

// L2 distance (default)
Image<Float32> dt = binary | DistanceTransform{};

// L1 distance, Mask5 precision
Image<Float32> dt_l1 = binary | DistanceTransform{}
    .dist_type(DistanceTransform::DistType::L1)
    .mask_size(DistanceTransform::MaskSize::Mask5);
```

`DistType`: `L1`, `L2` (default), `C` (Chebyshev).  
`MaskSize`: `Mask3` (default), `Mask5`, `Precise`.

### Watershed Segmentation Prep

Distance transform is the standard first step before watershed:

```cpp
Image<Gray>   binary = convert<Gray>(*src) | Threshold{}.mode(ThresholdMode::Otsu);
Image<Float32> dt    = binary | DistanceTransform{};

// Threshold the distance map to get sure-foreground seeds
cv::Mat dt_norm;
cv::normalize(dt.mat(), dt_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
Image<Gray> seeds{dt_norm};
Image<Gray> sure_fg = seeds | Threshold{}.value(0.6 * 255).mode(ThresholdMode::Binary);
```

## Morphological Shape Analysis

Morphological ops complement contour/component analysis for cleaning up binary masks.

```cpp
// Remove small noise blobs before finding contours
Image<Gray> clean = binary
    | MorphOpen{}.kernel_size(5)    // erode then dilate — removes small bright blobs
    | MorphClose{}.kernel_size(5);  // dilate then erode — fills small holes

// Highlight boundaries (morphological gradient = dilate - erode)
Image<Gray> edges = binary | MorphGradient{}.kernel_size(3);

// Isolate small bright features against a dark background
Image<Gray> bright_spots = binary | TopHat{}.kernel_size(15);

// Isolate small dark features against a bright background
Image<Gray> dark_spots   = binary | BlackHat{}.kernel_size(15);
```

## Adaptive Threshold

For images with uneven illumination, `AdaptiveThreshold` computes a per-pixel threshold from local neighbourhood statistics:

```cpp
Image<Gray> gray = convert<Gray>(*src);

// Gaussian-weighted local mean
Image<Gray> adaptive = gray | AdaptiveThreshold{}
    .method(AdaptiveThreshold::Method::Gaussian)
    .block_size(11)    // neighbourhood size (odd, ≥ 3; default: 11)
    .C(2.0);           // constant subtracted from mean (default: 2)

// Mean local method
Image<Gray> mean_thresh = gray | AdaptiveThreshold{}
    .method(AdaptiveThreshold::Method::Mean)
    .block_size(15);
```

## Full Analysis Pipeline Example

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"
#include "improc/visualization/visualization.hpp"
using namespace improc::core;
using namespace improc::io;
using namespace improc::visualization;

int main() {
    auto src = imread<BGR>("cells.png");
    if (!src) return 1;

    // Preprocess
    Image<Gray> gray   = convert<Gray>(*src);
    Image<Gray> binary = gray
        | GaussianBlur{}.kernel_size(5)
        | Threshold{}.mode(ThresholdMode::Otsu)
        | MorphOpen{}.kernel_size(3);

    // Label components
    ComponentMap cm = binary | ConnectedComponents{};
    std::cout << "Cells detected: " << (cm.count() - 1) << "\n";

    // Find contours and draw
    ContourSet cs = binary | FindContours{};
    Image<BGR> result = *src | DrawContours{cs}.color({0, 255, 0}).thickness(2);

    // Annotate each component with its area
    for (int i = 1; i < cm.count(); ++i) {
        auto [cx, cy] = cm.centroid(i);
        cv::putText(result.mat(), std::to_string(cm.area(i)),
                    {static_cast<int>(cx), static_cast<int>(cy)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }

    result | Show{"Analysis"};
    cv::waitKey(0);
}
```

## Next Steps

- [Feature Detection](feature-detection.md) — ORB, SIFT, AKAZE keypoints and matching
- [Building a Pipeline](building-a-pipeline.md) — all core ops at a glance
- [Segmentation](segmentation.md) — pixel-level classification with deep models
