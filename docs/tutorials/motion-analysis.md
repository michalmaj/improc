# Motion Analysis

This tutorial covers motion estimation in improc++: sparse feature tracking with Lucas-Kanade optical flow, dense per-pixel flow with Farneback and DIS, kernel-based object tracking with CamShift and MeanShift, and sub-pixel translation estimation with PhaseCorrelate. Use sparse methods when you only need to track specific feature points; use dense methods when you need the complete motion field for every pixel.

`Image<Flow>` is a `CV_32FC2` image — each pixel stores a `(dx, dy)` displacement vector in floating-point pixels.

## Prerequisites

- Completed [Building a Pipeline](building-a-pipeline.md)
- `improc/core/pipeline.hpp` (umbrella include for all core ops)
- `improc/io/image_io.hpp` for `imread`

## Sparse Lucas-Kanade Flow (`SparseLKFlow`)

Lucas-Kanade optical flow tracks a set of user-supplied points from one frame to the next using a pyramid of smoothed images. It is fast, accurate for small-to-medium displacements, and ideal when you already know which points matter — for example, corners detected by `cv::goodFeaturesToTrack`.

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"
using namespace improc::core;
using namespace improc::io;

int main() {
    auto frame1 = imread<BGR>("frame1.png");
    auto frame2 = imread<BGR>("frame2.png");
    if (!frame1 || !frame2) return 1;

    Image<Gray> gray1 = convert<Gray>(*frame1);
    Image<Gray> gray2 = convert<Gray>(*frame2);

    // Detect corners in frame1 to use as tracking seeds
    std::vector<cv::Point2f> pts1;
    cv::goodFeaturesToTrack(gray1.mat(), pts1, 200, 0.01, 10);

    SparseLKFlowResult res = SparseLKFlow{}
        .win_size({21, 21})  // search window per pyramid level (default: {21,21})
        .max_level(3)        // pyramid depth (default: 3)
        .max_iter(30)        // max Lucas-Kanade iterations (default: 30)
        .epsilon(0.01)       // convergence threshold (default: 0.01)
        (gray1, gray2, pts1);

    // status[i] = 1 if tracked, 0 if lost; points[i] = new position; error[i] = residual
    int n_tracked = 0;
    for (std::size_t i = 0; i < res.status.size(); ++i)
        if (res.status[i]) ++n_tracked;

    std::cout << "Tracked " << n_tracked << " / " << pts1.size() << " points\n";
}
```

`SparseLKFlowResult` has three parallel vectors, one entry per input point:

- `points` — new `cv::Point2f` positions in `frame2`
- `status` — `1` if the point was tracked successfully, `0` if it was lost (e.g., moved out of frame or had insufficient texture)
- `error` — per-point tracking residual; lower is better

Increase `win_size` and `max_level` when tracking large displacements; decrease them to reduce computation when motion is small.

## Dense Farneback Flow (`DenseFarnebackFlow`)

Farneback's algorithm approximates each image neighbourhood with a polynomial expansion and estimates the flow that minimises the difference between expansions. It produces an `Image<Flow>` covering every pixel — useful for motion segmentation, background subtraction, and action recognition.

```cpp
Image<Flow> flow = DenseFarnebackFlow{}
    .pyr_scale(0.5)    // pyramid scale factor (default: 0.5)
    .levels(3)         // pyramid levels (default: 3)
    .win_size(15)      // averaging window size (default: 15)
    .iterations(3)     // iterations per level (default: 3)
    .poly_n(5)         // polynomial neighbourhood (default: 5)
    .poly_sigma(1.2)   // Gaussian sigma for poly expansion (default: 1.2)
    (gray1, gray2);
```

Visualise the flow field as an HSV colour wheel — hue encodes direction, value encodes magnitude:

```cpp
// Visualize flow as HSV colour wheel: hue = direction, value = magnitude
cv::Mat planes[2];
cv::split(flow.mat(), planes);   // planes[0] = dx, planes[1] = dy

cv::Mat mag, ang;
cv::cartToPolar(planes[0], planes[1], mag, ang, true);  // true = degrees

cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
mag.convertTo(mag, CV_8U);
cv::Mat hue; ang.convertTo(hue, CV_8U, 255.0 / 360.0);
cv::Mat sat = cv::Mat::ones(mag.size(), CV_8U) * 255;

cv::Mat hsv, vis;
cv::merge(std::vector<cv::Mat>{hue, sat, mag}, hsv);
cv::cvtColor(hsv, vis, cv::COLOR_HSV2BGR);
cv::imshow("Flow", vis);
```

Larger `win_size` and more `levels` capture bigger displacements at the cost of blurring fine detail. Higher `poly_n` gives a smoother but less localised flow field.

## Dense DIS Flow (`DenseDISFlow`)

DIS (Dense Inverse Search) is significantly faster than Farneback for equal quality — often 5–10× — making it the preferred choice for real-time dense flow. Three presets balance speed against quality:

```cpp
// Three presets — quality ascending, speed descending
Image<Flow> dis_uf = DenseDISFlow{}.preset(DenseDISFlow::Preset::UltraFast)(gray1, gray2);
Image<Flow> dis_f  = DenseDISFlow{}.preset(DenseDISFlow::Preset::Fast)     (gray1, gray2);
Image<Flow> dis_m  = DenseDISFlow{}.preset(DenseDISFlow::Preset::Medium)   (gray1, gray2);
```

`UltraFast` runs at several hundred frames per second on HD images; `Medium` is the default and produces Farneback-quality results in a fraction of the time. The output `Image<Flow>` format is identical to Farneback's, so the HSV visualisation above applies unchanged.

## Kernel-Based Tracking / CamShift

CamShift (Continuously Adaptive MeanShift) tracks an object described by a colour histogram. It adapts the search window size and orientation at each frame, returning a `RotatedRect` that approximates the object's pose. The three-step process — build histogram, back-project, track — separates model building from tracking so you can update the histogram online.

### Build the colour model

```cpp
// Step 1 — build hue histogram from the object ROI in frame1
Image<BGR> hsv1 = bgr1 | ToHSV{};
cv::Mat hue1(hsv1.mat().rows, hsv1.mat().cols, CV_8U);
int from_to[] = {0, 0};
cv::mixChannels({hsv1.mat()}, {hue1}, from_to, 1);

cv::Mat hist;
int histSize = 32;
float range[] = {0.f, 180.f};
const float* histRange = {range};
cv::Rect roi{100, 80, 60, 60};
cv::Mat roi_hue = hue1(roi);
cv::calcHist(&roi_hue, 1, nullptr, cv::Mat{}, hist, 1, &histSize, &histRange);
cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX);
```

`ToHSV{}` converts the BGR image to HSV in-place (stored in an `Image<BGR>` wrapper, matching OpenCV's convention). We extract only the hue channel via `cv::mixChannels` because hue is more invariant to lighting changes than full colour.

### Back-project and track

```cpp
// Step 2 — back-project onto frame2
Image<BGR> hsv2 = bgr2 | ToHSV{};
cv::Mat hue2(hsv2.mat().rows, hsv2.mat().cols, CV_8U);
cv::mixChannels({hsv2.mat()}, {hue2}, from_to, 1);
cv::Mat bp_mat;
cv::calcBackProject(&hue2, 1, nullptr, hist, bp_mat, &histRange);
Image<Gray> bp{bp_mat};

// Step 3 — track
cv::Rect window = roi;
CamShiftResult res = CamShift{}
    .epsilon(1.0)    // convergence threshold (default: 1.0)
    .max_iter(10)    // max iterations (default: 10)
    (bp, window);    // window updated in-place to new position

cv::RotatedRect obj = res.object;   // tracked location and orientation
```

`CamShiftResult` contains a single field `object` — a `cv::RotatedRect` with `.center`, `.size`, and `.angle`. Pass `obj` to `cv::ellipse` to draw a tight oriented bounding box around the tracked region.

### MeanShift

MeanShift is the axis-aligned variant: the window does not rotate or resize, making it faster but less expressive. It returns the iteration count rather than a `RotatedRect`.

```cpp
cv::Rect window2 = roi;
int iters = MeanShift{}
    .epsilon(1.0)    // convergence threshold (default: 1.0)
    .max_iter(10)    // max iterations (default: 10)
    (bp, window2);   // window2 updated in-place; returns iteration count

std::cout << "Converged in " << iters << " iterations at " << window2 << "\n";
```

Both `CamShift` and `MeanShift` take `Image<Gray>` as the back-projection argument and mutate the `cv::Rect` window in-place.

## Sub-Pixel Translation (`PhaseCorrelate`)

Phase correlation computes the cross-power spectrum of two images in the frequency domain and finds the peak of the inverse DFT — giving a sub-pixel translation estimate in a single pass. It is ideal for camera stabilisation and image registration when the dominant motion is a global translation.

```cpp
// PhaseCorrelate requires Image<Float32> (single-channel float)
Image<Float32> f1 = convert<Gray>(*frame1) | ToFloat32{};
Image<Float32> f2 = convert<Gray>(*frame2) | ToFloat32{};

PhaseCorrelateResult pc = PhaseCorrelate{}(f1, f2);
std::cout << "Shift: (" << pc.shift.x << ", " << pc.shift.y << ") px\n";
std::cout << "Response: " << pc.response << "  (>0.3 = reliable)\n";
```

`PhaseCorrelateResult` fields:

- `shift` — `cv::Point2d` with sub-pixel `(dx, dy)` displacement
- `response` — confidence in the range 0..1; values above 0.3 indicate a reliable peak; values below suggest the frames contain little common structure or the shift exceeds the image's Nyquist limit

## Choosing a Method

| Op | Input | Output | Use case |
|---|---|---|---|
| `SparseLKFlow` | Gray pair + points | Points + status | Track specific features |
| `DenseFarnebackFlow` | Gray pair | `Image<Flow>` | Full-frame motion analysis |
| `DenseDISFlow` | Gray pair | `Image<Flow>` | Same, faster — prefer real-time |
| `CamShift` | Back-proj + rect | `RotatedRect` | Object tracking with colour model |
| `MeanShift` | Back-proj + rect | iteration count | Axis-aligned tracking |
| `PhaseCorrelate` | Float32 pair | shift + response | Camera stabilisation, translation |

## Next Steps

- [Camera Calibration](camera-calibration.md)
- [Object Detectors](object-detectors.md)
- [Building a Pipeline](building-a-pipeline.md)
