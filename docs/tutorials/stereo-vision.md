# Stereo Vision

This tutorial explains how to use two calibrated cameras to estimate depth, compute disparity maps, and reconstruct 3-D point clouds with improc++. It also covers the epipolar geometry tools — fundamental matrix, essential matrix, pose recovery, and triangulation — that are useful when you do not have a pre-calibrated stereo rig.

All operations live in `#include "improc/calib/pipeline.hpp"`.

## Prerequisites

- Completed [Camera Calibration](camera-calibration.md)
- `improc/calib/pipeline.hpp`

## The Stereo Geometry

Two cameras separated by a known horizontal baseline B observe the same 3-D point P. The point projects to pixel x_L in the left image and x_R in the right image. The **disparity** is d = x_L − x_R. Depth follows the thin-lens stereo formula:

```
Z = f · B / d
```

where f is the focal length (in pixels). Larger disparity means closer depth; infinite depth means zero disparity. After **rectification** both image rows are aligned, so matching is reduced to a 1-D horizontal search — this is the key step that makes disparity computation tractable.

## Stereo Calibration

`StereoCalibrate` estimates the relative pose (rotation R and translation T) between two cameras, refining their individual intrinsics at the same time. Collect paired chessboard views exactly as in the single-camera case, but detect corners in both images simultaneously:

```cpp
#include "improc/calib/pipeline.hpp"
using namespace improc::calib;
using namespace improc::core;

std::vector<std::vector<cv::Point3f>> all_obj;
std::vector<std::vector<cv::Point2f>> all_img1, all_img2;

for (auto& [left, right] : paired_frames) {
    auto r1 = left  | FindChessboardCornersSB{}.board_size({9, 6});
    auto r2 = right | FindChessboardCornersSB{}.board_size({9, 6});

    if (!r1.found || !r2.found) continue;   // both must succeed

    all_obj.push_back(make_chessboard_points({9, 6}, 0.025f));
    all_img1.push_back(r1.corners);
    all_img2.push_back(r2.corners);
}

// StereoCalibrate: joint optimisation of both cameras and their relative pose
StereoCalibrationResult sr = StereoCalibrate{}
    // Optionally seed with individual calibrations:
    // .K1(cal1.camera_matrix).dist1(cal1.dist_coeffs)
    // .K2(cal2.camera_matrix).dist2(cal2.dist_coeffs)
    // .flags(cv::CALIB_FIX_INTRINSIC)   // fix intrinsics, only solve for R, T
    (all_obj, all_img1, all_img2, image_size);
```

`StereoCalibrationResult` fields:

| Field | Type | Meaning |
|---|---|---|
| `K1`, `K2` | `cv::Mat` (3×3) | Left and right camera matrices |
| `dist1`, `dist2` | `cv::Mat` | Left and right distortion coefficients |
| `R` | `cv::Mat` (3×3) | Rotation from camera 1 to camera 2 |
| `T` | `cv::Mat` (3×1) | Translation from camera 1 to camera 2 (in metres if sq=0.025) |
| `E` | `cv::Mat` (3×3) | Essential matrix |
| `F` | `cv::Mat` (3×3) | Fundamental matrix |
| `rms` | `double` | RMS re-projection error across both cameras |

If you already have good individual calibrations, pass `cv::CALIB_FIX_INTRINSIC` to freeze K1, K2, dist1, dist2 and solve only for R and T — this converges faster and avoids over-parameterising the problem.

## Stereo Rectification

Rectification warps both images so that corresponding epipolar lines become horizontal and co-planar. After rectification, a point at column x in the left row r has its match somewhere on the same row r in the right image — a horizontal scan suffices.

```cpp
// StereoRectify: compute rectification transforms from stereo calibration results
StereoRectifyResult rr = StereoRectify{}
    .alpha(-1.0)   // -1 = keep all valid pixels (crops less); 0 = keep only fully valid
                   //  values between 0..1 trade off cropping vs. black borders
    (sr.K1, sr.dist1, sr.K2, sr.dist2, sr.R, sr.T, image_size);
```

`StereoRectifyResult` fields:

| Field | Type | Meaning |
|---|---|---|
| `R1`, `R2` | `cv::Mat` (3×3) | Rectification rotations for left and right |
| `P1`, `P2` | `cv::Mat` (3×4) | Projection matrices in the rectified coordinate system |
| `Q` | `cv::Mat` (4×4) | Disparity-to-depth reprojection matrix |
| `validROI1`, `validROI2` | `cv::Rect` | Valid pixel regions after rectification |

Apply rectification using `UndistortMap` and `Remap`, which combine undistortion and rectification in a single pass:

```cpp
// Left rectification maps: combine undistortion + rectification rotation
UndistortMapResult maps1 = UndistortMap{}
    .K(sr.K1)
    .dist(sr.dist1)
    .new_K(rr.P1)   // rectified projection matrix as new K
    .R(rr.R1)       // rectification rotation
    (image_size);

// Right rectification maps
UndistortMapResult maps2 = UndistortMap{}
    .K(sr.K2)
    .dist(sr.dist2)
    .new_K(rr.P2)
    .R(rr.R2)
    (image_size);

// Apply per frame — very fast, tables are pre-computed
Image<BGR> rect_left  = left_frame  | Remap{maps1.map1, maps1.map2};
Image<BGR> rect_right = right_frame | Remap{maps2.map1, maps2.map2};
```

After this step, horizontal scan lines in `rect_left` and `rect_right` are guaranteed to be epipolar lines.

## Computing Disparity

Disparity algorithms operate on grayscale images. improc++ wraps two OpenCV stereo matchers.

### StereoBM — fast block matching

`StereoBM` uses fixed-size block matching with a hardware-friendly sliding-window approach. It is the fastest option and works well for scenes with strong texture:

```cpp
Image<Gray> left_gray  = rect_left  | ToGray{};
Image<Gray> right_gray = rect_right | ToGray{};

cv::Mat disp_bm = StereoBM{}
    .num_disparities(64)   // disparity search range; must be divisible by 16 (default: 16)
    .block_size(15)        // matching block size; odd number in [5, 255] (default: 15)
    (left_gray, right_gray);
// Returns CV_16S; true disparity (in pixels) = value / 16.0
```

Larger `num_disparities` covers a wider depth range but is slower. Larger `block_size` is more robust to noise but blurs depth edges.

Convert to a visualisable 8-bit map:

```cpp
cv::Mat disp_vis;
disp_bm.convertTo(disp_vis, CV_8U, 255.0 / (16.0 * num_disparities));
```

### StereoSGBM — semi-global block matching

`StereoSGBM` (Semi-Global Block Matching) aggregates matching costs along multiple scanline paths. It produces significantly smoother and more complete disparity maps, at the cost of higher CPU usage:

```cpp
const int bsz = 5;
cv::Mat disp_sgbm = StereoSGBM{}
    .min_disparity(0)              // minimum disparity value (default: 0)
    .num_disparities(64)           // number of disparity levels (default: 64)
    .block_size(bsz)               // matched block size, must be odd (default: 3)
    .p1(8  * bsz * bsz)           // smoothness penalty for 1-pixel disparity change
    .p2(32 * bsz * bsz)           // smoothness penalty for larger changes (p2 > p1)
    .mode(cv::StereoSGBM::MODE_SGBM_3WAY)   // 3-way is faster than full SGBM
    (left_gray, right_gray);
// Returns CV_16S, same scaling as StereoBM: divide by 16 for float disparity
```

P1 and P2 are the SGBM smoothness penalties. The rule of thumb `P1 = 8·bsz²` and `P2 = 32·bsz²` gives good results on natural scenes.

## 3-D Reconstruction

With the disparity map and the Q matrix from `StereoRectify`, you can lift every pixel to a 3-D point:

```cpp
cv::Mat pts3d = ReprojectTo3D{}
    .handle_missing(false)   // false = include all points; true = skip invalid disparity
    (disp_sgbm, rr.Q);       // returns CV_32FC3: each pixel → (X, Y, Z) in metres
```

Filter out invalid points before use — pixels with zero or negative disparity produce infinity or NaN:

```cpp
for (int y = 0; y < pts3d.rows; ++y)
    for (int x = 0; x < pts3d.cols; ++x) {
        auto& p = pts3d.at<cv::Vec3f>(y, x);
        if (!std::isinf(p[2]) && p[2] > 0 && p[2] < max_depth) {
            // p[0], p[1], p[2]: X, Y, Z in the same units as the baseline
            point_cloud.push_back({p[0], p[1], p[2]});
        }
    }
```

The Q matrix encodes baseline, focal length, and principal point shift, so the recovered 3-D coordinates are in the same physical unit as the baseline (metres if you used 0.025 m squares).

## Epipolar Geometry

If you do not have a calibrated stereo rig — for example, when working with two uncalibrated images or tracking a moving camera — you can still recover the relative pose from feature point correspondences.

### Fundamental Matrix

The fundamental matrix F relates corresponding points in two uncalibrated images: for every pair (x1, x2), x2^T · F · x1 = 0.

```cpp
// FindFundamentalMat: RANSAC-based estimation from ≥8 point pairs
FundamentalMatResult fm = FindFundamentalMat{}
    .method(cv::FM_RANSAC)       // robust estimator (default)
    .ransac_threshold(3.0)       // reprojection threshold in pixels (default: 3.0)
    .confidence(0.99)            // required probability that result is correct
    (pts1, pts2);                // vector<cv::Point2f>, ≥8 matched pairs required
// fm.F: 3×3 CV_64F fundamental matrix
// fm.mask: CV_8U inlier flags (1 = inlier, 0 = outlier)
```

### Essential Matrix

When camera intrinsics K are known, the essential matrix E = K^T · F · K encodes the metric rotation and translation (up to scale):

```cpp
EssentialMatResult em = FindEssentialMat{}
    .method(cv::RANSAC)
    .threshold(1.0)     // reprojection threshold (default: 1.0)
    .confidence(0.99)
    (pts1, pts2, K);    // same K for both cameras (single-camera motion case)
// em.E: 3×3 CV_64F; em.mask: inlier flags
```

### Recovering Camera Pose

Decompose E into rotation and (unit) translation:

```cpp
// RecoverPose: decomposes E and selects the physically consistent solution
RecoverPoseResult rp = RecoverPose{}(em.E, pts1, pts2, K);
// rp.R: 3×3 rotation matrix
// rp.t: 3×1 unit translation (direction only, scale unknown)
// rp.inliers: count of points in front of both cameras
```

Translation is recovered only up to scale. To get metric depth you need a known distance in the scene, stereo baseline, or IMU data.

### Triangulating Points

Given two projection matrices P1 and P2, triangulate 3-D positions from matched 2-D points:

```cpp
// Build projection matrices from intrinsics and recovered pose
cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);         // left camera: no rotation/translation
cv::Mat P2_34 = cv::Mat::zeros(3, 4, CV_64F);
rp.R.copyTo(P2_34.colRange(0, 3));
rp.t.copyTo(P2_34.col(3));
cv::Mat P2 = K * P2_34;

cv::Mat hom = TriangulatePoints{}(P1, P2, pts1, pts2);
// Returns 4×N CV_32F in homogeneous coordinates
// Convert to Euclidean: X/W, Y/W, Z/W for each column
for (int i = 0; i < hom.cols; ++i) {
    float w = hom.at<float>(3, i);
    cv::Point3f pt3d(hom.at<float>(0, i) / w,
                     hom.at<float>(1, i) / w,
                     hom.at<float>(2, i) / w);
    // use pt3d ...
}
```

Triangulation accuracy degrades when the baseline angle between views is small (near-parallel cameras). For best results, choose well-separated viewpoints.

## Complete Example

See `examples/calib/demo_stereo.cpp` for a self-contained demo that builds a synthetic textured scene, computes disparity with both `StereoBM` and `StereoSGBM`, and reconstructs a 3-D point cloud with `ReprojectTo3D`.

## Next Steps

- [ArUco Markers](aruco-markers.md) — fiducial markers for pose estimation with a single camera
- [Camera Calibration](camera-calibration.md) — single-camera calibration workflow
- [Feature Detection](feature-detection.md) — keypoint matching for epipolar geometry inputs
