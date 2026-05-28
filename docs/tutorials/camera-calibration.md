# Camera Calibration

This tutorial covers how to calibrate a camera with improc++: detecting chessboard corners, collecting calibration data across multiple images, running the calibration solver, and undistorting frames. The whole workflow is driven by the `#include "improc/calib/pipeline.hpp"` umbrella header.

Real cameras deviate from the ideal pinhole model. The lens introduces radial and tangential distortion — straight lines appear curved, and the apparent image centre does not match the principal point. Calibration estimates the **camera matrix** K (focal lengths and principal point) and the **distortion coefficients** that describe these deviations. Once you have those, you can remove distortion from every subsequent frame, project 3-D points accurately onto the image plane, or recover metric depth.

## Prerequisites

- Completed [Building a Pipeline](building-a-pipeline.md)
- `improc/calib/pipeline.hpp` (includes `improc/core/pipeline.hpp` transitively)

## Detecting Chessboard Corners

A printed chessboard is the standard calibration target because its inner corner positions can be detected precisely and their ground-truth 3-D coordinates are known (they lie on a flat Z = 0 plane).

```cpp
#include "improc/calib/pipeline.hpp"
using namespace improc::calib;
using namespace improc::core;

Image<BGR> img = ...;   // load or capture a calibration frame

// FindChessboardCorners: classic Harris-corner detector, BGR auto-converted to Gray internally
FindChessboardResult r = img | FindChessboardCorners{}.board_size({9, 6});
// board_size = inner corners (cols, rows) — a 10×7 square board has 9×6 inner corners

if (r.found) {
    // r.corners: vector<cv::Point2f> — pixel coordinates of each inner corner
    std::cout << "Found " << r.corners.size() << " corners\n";
}
```

`FindChessboardCornersSB` is the newer, recommended alternative. It uses a different algorithm that provides built-in sub-pixel accuracy and is more robust to uneven lighting:

```cpp
// FindChessboardCornersSB: newer method with sub-pixel accuracy built in
FindChessboardResult r = img | FindChessboardCornersSB{}.board_size({9, 6});
// Prefer this over FindChessboardCorners for new code
```

When using the classic `FindChessboardCorners`, you can improve corner accuracy with an explicit sub-pixel refinement pass:

```cpp
// RefineCorners: sub-pixel refinement for corners found by FindChessboardCorners
// (not needed after FindChessboardCornersSB, which refines internally)
auto refined = RefineCorners{}
    .win_size(11)    // half-size of the search window in pixels (default: 11)
    .max_iter(30)    // maximum number of solver iterations (default: 30)
    .epsilon(0.001)  // convergence threshold (default: 0.001)
    (gray, r.corners);  // takes a Gray image and the initial corner list
// refined: vector<cv::Point2f> with sub-pixel positions
```

`win_size` controls the neighbourhood searched around each corner. Smaller values are more precise on dense patterns; larger values are more forgiving on blurry images.

## Collecting Calibration Data

Calibration requires the same physical corners to be observed from multiple viewpoints. The more views you collect — and the more you vary tilt, rotation, and distance — the better constrained the problem becomes. Aim for at least 10–15 views in a real workflow.

For each successful view you need a matching pair: the 3-D object points (the same every frame) and the 2-D image points (different every frame).

`make_chessboard_points` generates the Z = 0 object points for a flat board:

```cpp
// make_chessboard_points: generates inner-corner 3-D coordinates on the Z=0 plane
// board_size: inner corners (cols, rows)
// square_size: physical side length of one square in metres (or any consistent unit)
auto obj_pts_one = make_chessboard_points({9, 6}, 0.025f);
// Returns vector<cv::Point3f>:
//   (0, 0, 0), (0.025, 0, 0), (0.050, 0, 0), ..., (0.200, 0.125, 0)
```

Collecting across multiple frames:

```cpp
std::vector<std::vector<cv::Point3f>> all_obj;   // one entry per view
std::vector<std::vector<cv::Point2f>> all_img;   // one entry per view

for (const auto& frame : calibration_frames) {
    Image<BGR> img(frame);

    auto res = img | FindChessboardCornersSB{}.board_size({9, 6});
    if (!res.found) continue;   // skip frames where the board is not fully visible

    all_obj.push_back(make_chessboard_points({9, 6}, 0.025f));  // same every frame
    all_img.push_back(res.corners);                              // pixel positions this frame
}
std::cout << "Collected " << all_obj.size() << " valid views\n";
```

For best results, capture frames with the board:
- tilted ±20° in both X and Y
- at several distances (close, mid, far)
- in the corners and centre of the frame
- in good, even lighting without glare on the board surface

## Camera Calibration

Once you have at least three views (ideally 10+), pass the accumulated data to `CalibrateCamera`:

```cpp
// CalibrateCamera: non-linear least-squares solver (Levenberg–Marquardt)
CalibrationResult cal = CalibrateCamera{}
    .flags(0)   // 0 = estimate all parameters freely
                // or: cv::CALIB_FIX_PRINCIPAL_POINT, cv::CALIB_RATIONAL_MODEL, etc.
    (all_obj, all_img, image_size);
// image_size: cv::Size — pixel dimensions of all frames (must be the same)
```

`CalibrationResult` fields:

| Field | Type | Meaning |
|---|---|---|
| `camera_matrix` | `cv::Mat` (3×3 CV_64F) | Intrinsics: `[fx, 0, cx; 0, fy, cy; 0, 0, 1]` |
| `dist_coeffs` | `cv::Mat` (1×5 CV_64F) | `[k1, k2, p1, p2, k3]` |
| `rvecs` | `vector<cv::Mat>` | Rotation vector per view (Rodrigues) |
| `tvecs` | `vector<cv::Mat>` | Translation vector per view |
| `rms` | `double` | RMS re-projection error in pixels |

The **RMS re-projection error** measures how far projected 3-D corners deviate from the detected 2-D corners on average. A value below 1.0 px indicates a good calibration; above 2.0 px usually means poor coverage, blurry frames, or an incorrect board size.

```cpp
std::cout << "RMS: " << cal.rms << " px\n";
std::cout << "K:\n"  << cal.camera_matrix << "\n";
std::cout << "dist: " << cal.dist_coeffs.t() << "\n";
```

Common `flags` values:
- `0` — estimate all parameters, including both focal lengths independently
- `cv::CALIB_FIX_ASPECT_RATIO` — force `fx == fy`
- `cv::CALIB_FIX_PRINCIPAL_POINT` — fix the principal point at the image centre
- `cv::CALIB_RATIONAL_MODEL` — use a more accurate 8-parameter distortion model

## Undistorting Images

With calibration data in hand, you can remove distortion from any frame.

### Undistort pipeline op

The simplest approach: apply the op directly in a pipeline. Internally this builds the remap tables on every call, so it is best suited for single images or low-frame-rate scenarios.

```cpp
// Undistort: BGR-in, BGR-out pipeline op
Image<BGR> undist = img
    | Undistort{}.K(cal.camera_matrix).dist(cal.dist_coeffs);
// Preserves Image<BGR> semantics; underlying cv::undistort is called internally
```

### UndistortMap + Remap for video

When processing many frames (e.g. a live camera feed), computing the remap tables once and reusing them is far more efficient:

```cpp
// UndistortMap: pre-compute remap tables from calibration data — call once
UndistortMapResult maps = UndistortMap{}
    .K(cal.camera_matrix)     // 3×3 intrinsic matrix
    .dist(cal.dist_coeffs)    // distortion coefficients
    (image_size);
// maps.map1, maps.map2: cv::Mat remap tables (same size as image_size)

// Remap: apply pre-computed tables — call per frame, very fast
while (camera.grab(frame)) {
    Image<BGR> undist = frame | Remap{maps.map1, maps.map2};
    // process undist ...
}
```

`Remap` uses bilinear interpolation by default. The `UndistortMap` / `Remap` split matches OpenCV's own recommended pattern for video pipelines.

For stereo rectification, `UndistortMap` accepts optional `.new_K()` and `.R()` parameters that incorporate the rectification rotation from `StereoRectify`. See the [Stereo Vision](stereo-vision.md) tutorial for that workflow.

## Saving and Loading Calibration

`CalibrateCamera` does not write files — that is handled directly with OpenCV's `cv::FileStorage`. Save after calibration and load before using the camera in production:

```cpp
// Save
{
    cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << cal.camera_matrix;
    fs << "dist_coeffs"   << cal.dist_coeffs;
    fs << "rms"           << cal.rms;
}

// Load
cv::Mat K, dist;
{
    cv::FileStorage fs("calibration.yml", cv::FileStorage::READ);
    fs["camera_matrix"] >> K;
    fs["dist_coeffs"]   >> dist;
}

// Use loaded values exactly as you would the CalibrationResult fields
Image<BGR> undist = frame | Undistort{}.K(K).dist(dist);
```

The YAML format is human-readable and easy to inspect. Store at least `camera_matrix`, `dist_coeffs`, and `rms` so you can verify calibration quality when loading old files.

## Complete Example

See `examples/calib/demo_calibrate.cpp` for a self-contained demo that synthesises five chessboard views, runs calibration, and shows both undistortion approaches.

## Next Steps

- [Stereo Vision](stereo-vision.md) — two-camera depth estimation and epipolar geometry
- [ArUco Markers](aruco-markers.md) — fiducial markers for pose estimation and board calibration
- [Feature Detection](feature-detection.md) — keypoint detection and descriptor matching
