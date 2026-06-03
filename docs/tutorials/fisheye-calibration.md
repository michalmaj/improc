# Fisheye Camera Calibration

Fisheye lenses use a fundamentally different projection model from the standard pinhole camera. A conventional lens projects approximately linearly up to ±30–40° from the optical axis. A fisheye lens deliberately introduces extreme barrel distortion to project fields of view up to 180° or more onto the sensor — something the standard radial-tangential distortion model cannot represent accurately even with many coefficients. OpenCV's `cv::fisheye` module implements the Kannala–Brandt model, which describes the relationship between the angle of incidence and the distorted image radius with four equidistant coefficients (k1, k2, k3, k4). This model is more accurate for wide-angle lenses and is what all six `improc++` fisheye ops use internally.

When to use the fisheye model instead of the standard pinhole model:
- Field of view greater than ~100° (typical of action cameras, surveillance wide-angles, automotive cameras)
- The standard calibration produces visibly curved undistorted images or a high RMS error even with many views
- You are working with catadioptric (mirror-based) or omnidirectional cameras

When the standard model is better:
- Normal photography and robotics lenses with FoV below ~90°
- You need to exchange calibration data with software that does not support the fisheye model

All fisheye ops are in `improc::calib` and live in `include/improc/calib/ops/fisheye.hpp`, included automatically by `improc/calib/pipeline.hpp`.

## Prerequisites

- Completed [Camera Calibration](camera-calibration.md)
- `improc/calib/pipeline.hpp`

## Collecting Calibration Data

The data-collection step is identical to the standard workflow — detect chessboard corners and accumulate point correspondences across views. The difference is that you need more angular coverage because fisheye lenses distort radially rather than in a localised region. Tilt the board aggressively (up to ±45°) and push it close to the image edges.

```cpp
#include "improc/calib/pipeline.hpp"
using namespace improc::calib;
using namespace improc::core;

std::vector<std::vector<cv::Point3f>> all_obj;
std::vector<std::vector<cv::Point2f>> all_img;

for (const auto& frame : calibration_frames) {
    Image<BGR> img(frame);
    auto res = img | FindChessboardCornersSB{}.board_size({9, 6});
    if (!res.found) continue;

    all_obj.push_back(make_chessboard_points({9, 6}, 0.025f));
    all_img.push_back(res.corners);
}
std::cout << "Collected " << all_obj.size() << " valid views\n";
// Aim for 15-20 views; include board near image corners and at steep angles
```

For a fisheye lens, fill the entire frame — the outer region carries the most distortion information and is underfit if you only image the board near the centre.

## Calibrating a Fisheye Camera

Pass the collected correspondences to `FisheyeCalibrate`:

```cpp
// FisheyeCalibrate: Kannala-Brandt fisheye calibration solver
CalibrationResult cal = FisheyeCalibrate{}
    .flags(cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
           cv::fisheye::CALIB_CHECK_COND          |
           cv::fisheye::CALIB_FIX_SKEW)           // default flags
    (all_obj, all_img, image_size);
```

The return type is the same `CalibrationResult` as standard calibration, with one critical difference: **`dist_coeffs` contains four equidistant coefficients [k1, k2, k3, k4]**, not the five standard Plumb-Bob coefficients [k1, k2, p1, p2, k3]. Do not mix these with standard distortion parameters.

| Field | Type | Meaning |
|---|---|---|
| `camera_matrix` | `cv::Mat` (3×3 CV_64F) | Intrinsics: `[fx, 0, cx; 0, fy, cy; 0, 0, 1]` |
| `dist_coeffs` | `cv::Mat` (4×1 CV_64F) | Kannala–Brandt: `[k1, k2, k3, k4]` |
| `rvecs` | `vector<cv::Mat>` | Per-view rotation vectors |
| `tvecs` | `vector<cv::Mat>` | Per-view translation vectors |
| `rms` | `double` | RMS re-projection error in pixels |

```cpp
std::cout << "RMS: " << cal.rms << " px\n";
// A good fisheye calibration typically achieves < 1.0 px RMS
// Values above 1.5 px suggest insufficient angular coverage or blurry frames
```

Common `FisheyeCalibrate` flags:

| Flag | Effect |
|---|---|
| `cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC` | Re-estimates per-view poses at each iteration — almost always needed |
| `cv::fisheye::CALIB_CHECK_COND` | Rejects frames that make the system ill-conditioned; recommended |
| `cv::fisheye::CALIB_FIX_SKEW` | Fixes skew to zero — appropriate for modern sensors |
| `cv::fisheye::CALIB_FIX_K1` … `CALIB_FIX_K4` | Fix individual distortion coefficients |

## Undistorting Images

### Pipeline op (single image)

`FisheyeUndistort` is a templated pipeline op — it accepts any `Image<F>` and returns the same format undistorted. Because the fisheye projection folds the extreme angles inward, the result typically shows black borders; pass a scaled `new_K` to zoom in and fill the frame.

```cpp
// FisheyeUndistort: single-image undistortion — BGR-in, BGR-out (or any format)
Image<BGR> undist = img
    | FisheyeUndistort{}
        .K(cal.camera_matrix)
        .dist(cal.dist_coeffs);
// Optional: supply new_K to crop/scale the output field of view
// A common choice is to scale the focal lengths by 0.5 to preserve more area:
cv::Mat new_K = cal.camera_matrix.clone();
new_K.at<double>(0, 0) *= 0.5;  // fx
new_K.at<double>(1, 1) *= 0.5;  // fy
Image<BGR> zoomed_out = img
    | FisheyeUndistort{}
        .K(cal.camera_matrix)
        .dist(cal.dist_coeffs)
        .new_K(new_K);
```

### Pre-computed maps for video

For live streams, computing maps once and reusing them is far more efficient. `FisheyeInitRectifyMap` produces the same `UndistortMapResult` as the standard `UndistortMap`, so the same `Remap` op applies both.

```cpp
// FisheyeInitRectifyMap: compute maps once
UndistortMapResult maps = FisheyeInitRectifyMap{}
    .K(cal.camera_matrix)
    .dist(cal.dist_coeffs)
    // .new_K(new_K)   // optional: scale output FoV as above
    // .R(R)           // optional: rectification rotation (for stereo rigs)
    (image_size);

// Remap: apply maps per frame — very fast
while (camera.getFrame()) {
    auto frame = camera.getFrame();
    if (!frame || !frame->rgb) continue;
    Image<BGR> undist = *frame->rgb | Remap{maps.map1, maps.map2};
    // process undist ...
}
```

## Undistorting Point Sets

To convert pixel coordinates detected on a fisheye image into normalised undistorted coordinates (for use with geometry algorithms that assume a pinhole model), use `FisheyeUndistortPoints`:

```cpp
// FisheyeUndistortPoints: normalise fisheye image points to the ideal plane
std::vector<cv::Point2f> distorted = { /* detected pixel coordinates */ };

std::vector<cv::Point2f> undistorted = FisheyeUndistortPoints{}
    .K(cal.camera_matrix)
    .dist(cal.dist_coeffs)
    // .R(R)  // optional rectification rotation
    // .P(P)  // optional new projection matrix; omit to get normalised coordinates
    (distorted);
// Without .P(), output is in normalised camera coordinates (z=1 plane)
// With .P(new_K), output is in pixel coordinates for the undistorted view
```

`FisheyeUndistortPoints` without `.P()` is the right pre-processing step before passing points to `FindEssentialMat` or `RecoverPose`.

## Saving and Loading

Use `cv::FileStorage` directly — the format is identical to standard calibration, but the distortion vector is 4×1, not 5×1:

```cpp
// Save
{
    cv::FileStorage fs("fisheye_calibration.yml", cv::FileStorage::WRITE);
    fs << "camera_matrix" << cal.camera_matrix;
    fs << "dist_coeffs"   << cal.dist_coeffs;   // 4×1: [k1, k2, k3, k4]
    fs << "rms"           << cal.rms;
}

// Load
cv::Mat K, D;
{
    cv::FileStorage fs("fisheye_calibration.yml", cv::FileStorage::READ);
    fs["camera_matrix"] >> K;
    fs["dist_coeffs"]   >> D;   // must be 4×1 — do not pass to standard Undistort
}
```

## Stereo Fisheye Rig

### Stereo Calibration

`FisheyeStereoCalibrate` jointly estimates the intrinsics and the relative pose (R, T) between two fisheye cameras. Provide initial guesses for both cameras — run `FisheyeCalibrate` on each individually first and pass the results in.

```cpp
// Step 1: calibrate each camera independently
CalibrationResult left_cal  = FisheyeCalibrate{}(all_obj, left_pts,  image_size);
CalibrationResult right_cal = FisheyeCalibrate{}(all_obj, right_pts, image_size);

// Step 2: jointly calibrate the stereo rig
StereoCalibrationResult rig = FisheyeStereoCalibrate{}
    .K1(left_cal.camera_matrix) .dist1(left_cal.dist_coeffs)
    .K2(right_cal.camera_matrix).dist2(right_cal.dist_coeffs)
    .flags(cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC |
           cv::fisheye::CALIB_CHECK_COND           |
           cv::fisheye::CALIB_FIX_SKEW             |
           cv::fisheye::CALIB_FIX_INTRINSIC)       // fix individual intrinsics
    (all_obj, left_pts, right_pts, image_size);

std::cout << "Stereo RMS: " << rig.rms << " px\n";
```

> **Note:** `StereoCalibrationResult.E` and `.F` (essential and fundamental matrices) are **not populated** by `FisheyeStereoCalibrate`. This is a limitation of the `cv::fisheye` API — it does not compute E or F. Use `rig.R` and `rig.T` for the inter-camera geometry instead.

| Field | Type | Meaning |
|---|---|---|
| `rig.K1`, `rig.K2` | `cv::Mat` (3×3 CV_64F) | Refined camera intrinsics |
| `rig.dist1`, `rig.dist2` | `cv::Mat` (4×1 CV_64F) | Refined [k1, k2, k3, k4] per camera |
| `rig.R` | `cv::Mat` (3×3 CV_64F) | Rotation from left to right camera |
| `rig.T` | `cv::Mat` (3×1 CV_64F) | Translation from left to right camera |
| `rig.E`, `rig.F` | `cv::Mat` | **Not computed** — will be empty |

### Stereo Rectification

`FisheyeStereoRectify` computes the rectification transforms and projection matrices that remap both camera images onto a common fronto-parallel plane, enabling standard disparity computation.

```cpp
// FisheyeStereoRectify: compute rectification transforms
StereoRectifyResult rect = FisheyeStereoRectify{}
    .image_size(image_size)
    .flags(cv::CALIB_ZERO_DISPARITY)  // align principal points (default)
    .balance(0.0)                     // 0.0 = tightest valid crop; 1.0 = full FoV
    .fov_scale(1.0)                   // additional FoV scaling (default: 1.0)
    (rig.K1, rig.dist1, rig.K2, rig.dist2, rig.R, rig.T);

// rect.R1, rect.R2: 3×3 rectification rotations for left and right
// rect.P1, rect.P2: 3×4 projection matrices in rectified space
// rect.Q:           4×4 disparity-to-depth matrix
// rect.validROI1, rect.validROI2: valid pixel regions after rectification
```

The `balance` parameter controls the trade-off between field of view and valid pixel area after rectification. A value of `0.0` crops to only the pixels visible in both cameras (no black borders); `1.0` preserves all pixels from both cameras (more black borders, larger total FoV).

### Building the Remap Tables

Use `FisheyeInitRectifyMap` with the rectification outputs to pre-compute per-camera remap tables:

```cpp
// Left camera
UndistortMapResult left_maps = FisheyeInitRectifyMap{}
    .K(rig.K1).dist(rig.dist1)
    .R(rect.R1).new_K(rect.P1(cv::Rect(0, 0, 3, 3)))
    (image_size);

// Right camera
UndistortMapResult right_maps = FisheyeInitRectifyMap{}
    .K(rig.K2).dist(rig.dist2)
    .R(rect.R2).new_K(rect.P2(cv::Rect(0, 0, 3, 3)))
    (image_size);

// Apply per frame
Image<BGR> left_rect  = left_frame  | Remap{left_maps.map1,  left_maps.map2};
Image<BGR> right_rect = right_frame | Remap{right_maps.map1, right_maps.map2};
// After rectification: horizontal scan-line correspondence holds for disparity
```

## Fisheye vs Standard: Quick Comparison

| Aspect | Standard (`CalibrateCamera`) | Fisheye (`FisheyeCalibrate`) |
|---|---|---|
| Distortion model | Plumb-Bob: k1, k2, p1, p2, k3 | Kannala–Brandt: k1, k2, k3, k4 |
| `dist_coeffs` shape | 1×5 or 1×8+ | 4×1 |
| Suitable FoV | Up to ~90° | Up to 180°+ |
| Stereo E/F matrices | Computed | Not computed (cv::fisheye limitation) |
| Undistort op | `Undistort{}` | `FisheyeUndistort{}` |
| Map op | `UndistortMap{}` | `FisheyeInitRectifyMap{}` |

## Next Steps

- [Camera Calibration](camera-calibration.md) — standard pinhole calibration
- [Stereo Vision](stereo-vision.md) — depth estimation from a calibrated stereo rig
- [ArUco Markers](aruco-markers.md) — pose estimation with fiducial markers
