# ArUco Markers

This tutorial explains how to generate, detect, and estimate the pose of ArUco and ChArUco markers with improc++. ArUco markers are square binary patterns that can be identified and localised in a single camera image without any additional reference object. They are widely used in robotics, augmented reality, and camera calibration.

All operations are in `#include "improc/calib/pipeline.hpp"`.

## Prerequisites

- Completed [Camera Calibration](camera-calibration.md)
- `improc/calib/pipeline.hpp`

## What Are ArUco Markers?

An ArUco marker is a square image with a black border and a binary grid inside. Each pattern in a **dictionary** has a unique ID. Because the four corners of a square marker are always visible and geometrically well-defined, a single marker gives you all the information needed to estimate a camera's pose relative to the marker — as long as you know the physical side length and camera intrinsics.

**ChArUco boards** combine an ArUco dictionary with a standard chessboard: ArUco markers are embedded inside chessboard squares. This gives you the best of both worlds — the reliable ID detection of ArUco and the sub-pixel corner accuracy of chessboard corners. ChArUco boards are the recommended target for high-precision camera calibration.

Common use cases:
- Robot arm end-effector tracking
- Augmented reality object placement
- Camera-to-world extrinsic calibration
- Stereo rig hand-eye calibration

## Dictionaries

A dictionary defines the set of valid marker patterns. Use `ArucoDict` to obtain one:

```cpp
#include "improc/calib/pipeline.hpp"
using namespace improc::calib;

// ArucoDict: returns a cv::aruco::Dictionary for the requested predefined type
cv::aruco::Dictionary dict = ArucoDict{}(cv::aruco::DICT_4X4_50);
// DICT_4X4_50: 50 unique markers on a 4×4 bit grid — small, fast, low capacity
```

Commonly used dictionary types:

| Dictionary | Grid | Markers | Notes |
|---|---|---|---|
| `DICT_4X4_50` | 4×4 | 50 | Fast detection, small set |
| `DICT_4X4_250` | 4×4 | 250 | Larger set, same grid |
| `DICT_5X5_100` | 5×5 | 100 | More robust to printing errors |
| `DICT_6X6_250` | 6×6 | 250 | Good balance of size and robustness |
| `DICT_ARUCO_ORIGINAL` | 5×5 | 1024 | Original ArUco library dictionary |

Choose the smallest dictionary that covers your ID range — smaller grids are detected more reliably at a distance.

## Generating Markers

`GenerateAruco` renders a single marker as a `Image<Gray>`:

```cpp
// GenerateAruco: renders marker ID from the dictionary as a square grayscale image
Image<Gray> marker = GenerateAruco{}
    .border_bits(1)    // white border width in bit-cell units (default: 1)
    (dict, 7, 200);    // dict, marker ID, output side in pixels
// Returns Image<Gray>: 200×200 px square including the white border
```

The `border_bits` parameter controls the quiet zone — a white border surrounding the pattern that aids detection. One bit cell at standard resolution is usually sufficient. For small printed markers, increase to 2 for more reliable detection.

To save a marker for printing:

```cpp
// cv::imwrite works directly with .mat()
cv::imwrite("marker_id7.png", marker.mat());

// Or display it
cv::imshow("Marker ID 7", marker.mat());
cv::waitKey(0);
```

Print at a known physical size. If you print a 200 px image at 200 dpi, each cell is 1 mm. Record the physical side length — you will need it for pose estimation.

## Detecting Markers

`DetectAruco` finds all markers from the dictionary in a BGR scene image:

```cpp
Image<BGR> scene = ...;

// DetectAruco: searches the whole image for markers from the given dictionary
ArucoResult res = DetectAruco{}(scene, dict);

std::cout << "Found " << res.ids.size() << " marker(s)\n";
for (std::size_t i = 0; i < res.ids.size(); ++i)
    std::cout << "  ID " << res.ids[i] << "\n";
```

`ArucoResult` fields:

| Field | Type | Meaning |
|---|---|---|
| `ids` | `vector<int>` | ID of each detected marker |
| `corners` | `vector<vector<cv::Point2f>>` | Four corner points per marker (clockwise from top-left) |
| `rejected` | `vector<vector<cv::Point2f>>` | Candidate regions that failed ID decoding |

Corner order is always: top-left, top-right, bottom-right, bottom-left. This convention lets you determine marker orientation unambiguously.

### Drawing Detected Markers

`DrawAruco` has two overloads. The first draws corner outlines and IDs:

```cpp
// DrawAruco overload 1: draw corner outlines and ID text on a BGR image
cv::Mat annotated = DrawAruco{}(scene.mat().clone(), res);
// Pass .clone() to avoid modifying the original image
cv::imshow("Detected markers", annotated);
```

This is useful for verifying detection during development — you can see which markers were found and whether their corners are accurate.

## Marker Pose Estimation

With a calibrated camera, you can recover the 3-D pose of each detected marker relative to the camera:

```cpp
// Load or use calibration results from CalibrateCamera
cv::Mat K    = cal.camera_matrix;
cv::Mat dist = cal.dist_coeffs;

float marker_length = 0.05f;   // physical side length in metres (must match your print)

// ArucoPose: estimates 6-DOF pose for each detected marker
std::vector<ArucoPoseResult> poses = ArucoPose{}(res, K, dist, marker_length);
```

`ArucoPoseResult` fields per marker:

| Field | Type | Meaning |
|---|---|---|
| `id` | `int` | Marker ID |
| `rvec` | `cv::Mat` (3×1) | Rotation vector (Rodrigues) — axis × angle |
| `tvec` | `cv::Mat` (3×1) | Translation vector in metres (camera frame) |

`tvec` gives the position of the marker origin (its centre) in the camera coordinate system. The Z component is the distance along the optical axis.

Convert a rotation vector to a rotation matrix when needed:

```cpp
cv::Mat R;
cv::Rodrigues(poses[0].rvec, R);   // 3×3 rotation matrix
```

Draw 3-D coordinate axes on each marker to visualise pose:

```cpp
// DrawAruco overload 2: add 3-D axis frames per detected marker
cv::Mat with_axes = DrawAruco{}
    .axis_length(0.03f)    // axis length in world units (default: 0.05)
    (scene.mat().clone(), res, poses, K, dist);
// Draws X (red), Y (green), Z (blue) axes originating from each marker centre
cv::imshow("Marker poses", with_axes);
```

The axes confirm orientation: Z points out of the marker face toward the camera; X and Y follow the marker plane.

## ChArUco Boards

A ChArUco board provides more accurate calibration than either a plain chessboard or individual ArUco markers alone. The chessboard sub-pixel corner accuracy makes it suitable for high-quality single-camera and stereo calibration; the ArUco markers handle partial occlusion gracefully (a plain chessboard fails if any square is covered).

```cpp
Image<BGR> scene = ...;

// CharucoBoard: detects both ArUco markers and chessboard corners in the scene
CharucoResult cr = CharucoBoard{}
    .board_size({5, 7})       // chessboard inner corners (cols, rows)
    .square_length(0.04f)     // chessboard square side in metres
    .marker_length(0.02f)     // ArUco marker side in metres (must be ≤ square_length)
    (scene, dict);
```

`CharucoResult` fields:

| Field | Type | Meaning |
|---|---|---|
| `charuco_corners` | `vector<cv::Point2f>` | Sub-pixel chessboard corner positions |
| `charuco_ids` | `vector<int>` | ID of each detected chessboard corner |
| `marker_corners` | `vector<vector<cv::Point2f>>` | Corner points of each ArUco marker |
| `marker_ids` | `vector<int>` | ID of each detected ArUco marker |

`charuco_corners` and `charuco_ids` are what you pass to a calibration solver. Unlike plain chessboard detection, partial visibility is fine — corners that are visible alongside their neighbouring ArUco markers are still usable. This makes ChArUco calibration practical with handheld boards where the edges are sometimes cropped.

Collect ChArUco frames for calibration the same way as with a plain chessboard — capture at multiple angles, distances, and positions across the frame, then pass the accumulated `charuco_corners` as image points and the matching 3-D object points to `CalibrateCamera`.

## Complete Example

See `examples/calib/demo_aruco.cpp` for a self-contained demo that generates a marker, embeds it in a synthetic scene, runs detection, and visualises both corner outlines and 3-D pose axes.

## Next Steps

- [Camera Calibration](camera-calibration.md) — single-camera intrinsic calibration with chessboards
- [Stereo Vision](stereo-vision.md) — two-camera depth estimation
- [Feature Detection](feature-detection.md) — alternative keypoint-based localisation
