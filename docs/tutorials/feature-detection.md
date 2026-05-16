# Feature Detection and Matching

This tutorial shows how to detect keypoints, compute descriptors, and match features between images using improc++. Topics covered: ORB, SIFT, AKAZE detectors and descriptors; brute-force and FLANN matching; and visualizing keypoints and matches.

## Prerequisites

- Completed [Building a Pipeline](building-a-pipeline.md)
- `improc/core/pipeline.hpp` (includes all feature ops)

## Detect Keypoints

All detectors accept `Image<Gray>` and return a `KeypointSet`. Use the pipeline `operator|` or call directly.

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"

using namespace improc::core;
using namespace improc::io;

int main() {
    auto src = imread<BGR>("scene.png");
    if (!src) return 1;
    auto gray = convert<Gray>(*src);

    // ORB — fast binary descriptor, good for real-time matching
    KeypointSet orb_kps = gray | DetectORB{}
        .max_features(500)    // max keypoints to retain (default: 500)
        .scale_factor(1.2f)   // image pyramid scale factor
        .n_levels(8);         // number of pyramid levels

    // SIFT — scale- and rotation-invariant, slower, CV_32F descriptors
    KeypointSet sift_kps = gray | DetectSIFT{}
        .max_features(0)        // 0 = no limit (default)
        .n_octave_layers(3);    // layers per octave

    // AKAZE — fast, binary descriptors, robust to affine transformations
    KeypointSet akaze_kps = gray | DetectAKAZE{}
        .threshold(0.001f);     // response threshold (default: 0.001)

    std::cout << "ORB:   " << orb_kps.size()   << " keypoints\n";
    std::cout << "SIFT:  " << sift_kps.size()  << " keypoints\n";
    std::cout << "AKAZE: " << akaze_kps.size() << " keypoints\n";
}
```

`KeypointSet` wraps `std::vector<cv::KeyPoint>`. Access individual keypoints via `.keypoints[i]`.

## Compute Descriptors

`Describe*` ops take a `KeypointSet` and the image, and return a `DescriptorSet` (keypoints + `cv::Mat` descriptors).

```cpp
// ORB descriptors (CV_8U, 32 bytes/keypoint)
DescriptorSet orb_desc = gray | DescribeORB{orb_kps};

// SIFT descriptors (CV_32F, 128 floats/keypoint)
DescriptorSet sift_desc = gray | DescribeSIFT{sift_kps};

// AKAZE descriptors (CV_8U)
DescriptorSet akaze_desc = gray | DescribeAKAZE{akaze_kps};

// Both accept Image<Gray> or Image<BGR> (converted internally)
DescriptorSet from_bgr = *src | DescribeORB{orb_kps};
```

`DescriptorSet` fields: `.keypoints` (the `KeypointSet` used for description), `.descriptors` (the raw `cv::Mat`).

## Detect + Describe in One Pass

The common pattern chains detection and description:

```cpp
auto gray1 = convert<Gray>(img1);
auto gray2 = convert<Gray>(img2);

auto desc1 = gray1 | DescribeORB{ gray1 | DetectORB{}.max_features(300) };
auto desc2 = gray2 | DescribeORB{ gray2 | DetectORB{}.max_features(300) };
```

## Match Descriptors

### Brute-Force Matching (`MatchBF`)

Norm type is auto-detected: `NORM_HAMMING` for binary (ORB/AKAZE) descriptors, `NORM_L2` for float (SIFT).

```cpp
// Basic match
MatchSet ms = MatchBF{desc1, desc2}();

// With cross-check (returns only symmetric matches — usually fewer, higher quality)
MatchSet ms_sym = MatchBF{desc1, desc2}
    .cross_check(true)
    ();

// With a distance filter (drop matches with distance > threshold)
MatchSet ms_close = MatchBF{desc1, desc2}
    .max_distance(80.0f)
    ();

std::cout << "Matches: " << ms.matches.size() << "\n";
for (const auto& m : ms.matches)
    std::cout << "query=" << m.queryIdx << " train=" << m.trainIdx
              << " dist=" << m.distance << "\n";
```

### FLANN Matching with Lowe Ratio Test (`MatchFlann`)

`MatchFlann` runs kNN(k=2) and keeps matches where `d1 < ratio_threshold * d2`. Works only with CV_32F descriptors (SIFT); throws `ParameterError` for binary descriptors.

```cpp
// SIFT descriptors required
MatchSet ms = MatchFlann{sift_desc1, sift_desc2}
    .ratio_threshold(0.75f)   // Lowe ratio (default: 0.7)
    ();

std::cout << "Good matches after ratio test: " << ms.size() << "\n";
```

## Visualize Keypoints

`DrawKeypoints` draws keypoints with `DRAW_RICH_KEYPOINTS` (oriented, scaled circles). Accepts `Image<Gray>` or `Image<BGR>`, always returns `Image<BGR>`.

```cpp
#include "improc/visualization/visualization.hpp"
using namespace improc::visualization;

Image<BGR> vis = gray | DrawKeypoints{orb_kps};
vis | Show{"ORB Keypoints"};
```

## Visualize Matches

`DrawMatches` is a callable (not a pipeline op) that renders two images side-by-side with connecting lines for each match.

```cpp
Image<BGR> vis = DrawMatches{
    img1, desc1.keypoints,
    img2, desc2.keypoints,
    ms
}();

vis | Show{"Matches"};
imwrite("matches.png", vis);
```

Output width = `img1.cols + img2.cols`; height = `max(img1.rows, img2.rows)`.

## Complete Example — Image Stitching Preparation

```cpp
#include "improc/core/pipeline.hpp"
#include "improc/io/image_io.hpp"
#include "improc/visualization/visualization.hpp"
using namespace improc::core;
using namespace improc::io;
using namespace improc::visualization;

int main() {
    auto img1 = imread<BGR>("left.png");
    auto img2 = imread<BGR>("right.png");
    if (!img1 || !img2) return 1;

    auto g1 = convert<Gray>(*img1);
    auto g2 = convert<Gray>(*img2);

    // Detect + describe with SIFT
    auto kps1 = g1 | DetectSIFT{}.max_features(500);
    auto kps2 = g2 | DetectSIFT{}.max_features(500);
    auto desc1 = g1 | DescribeSIFT{kps1};
    auto desc2 = g2 | DescribeSIFT{kps2};

    // Match with Lowe ratio test
    MatchSet ms = MatchFlann{desc1, desc2}.ratio_threshold(0.75f)();
    std::cout << "Good matches: " << ms.size() << "\n";

    // Visualize
    DrawMatches{*img1, kps1, *img2, kps2, ms}()
        | Show{"SIFT Matches"};

    // Compute homography from matches (from improc::core::find_homography)
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : ms.matches) {
        pts1.push_back(kps1.keypoints[m.queryIdx].pt);
        pts2.push_back(kps2.keypoints[m.trainIdx].pt);
    }
    auto H = find_homography(pts1, pts2);  // returns cv::Mat

    cv::waitKey(0);
}
```

## Choosing a Detector

| Detector | Descriptor type | Speed | Invariance | Use case |
|---|---|---|---|---|
| `DetectORB` | Binary (CV_8U, 32B) | Fast | Rotation, scale | Real-time, mobile |
| `DetectSIFT` | Float (CV_32F, 128B) | Slow | Rotation, scale, affine | High-accuracy matching |
| `DetectAKAZE` | Binary (CV_8U) | Medium | Affine, non-linear | Robust to distortion |

Use `MatchBF` with `cross_check(true)` for ORB/AKAZE; use `MatchFlann` with `ratio_threshold` for SIFT.

## Next Steps

- [Image Analysis](image-analysis.md) — contours, connected components, distance transform
- [Building a Pipeline](building-a-pipeline.md) — core ops and format conversions
- [ML Inference](ml-inference.md) — learning-based feature matching
