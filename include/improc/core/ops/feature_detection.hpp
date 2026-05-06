// include/improc/core/ops/feature_detection.hpp
#pragma once
#include <vector>
#include <cstddef>
#include <opencv2/features2d.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Result of a feature-detection operation.
 *
 * Plain struct with a public `keypoints` vector. Use `size()` and `empty()`
 * for convenience; iterate `keypoints` directly for per-keypoint access.
 *
 * @code
 * KeypointSet ks = gray | DetectORB{};
 * for (auto& kp : ks.keypoints)
 *     std::cout << kp.pt << '\n';
 * @endcode
 */
struct KeypointSet {
    std::vector<cv::KeyPoint> keypoints;

    std::size_t size()  const { return keypoints.size(); }
    bool        empty() const { return keypoints.empty(); }
};

/**
 * @brief Pipeline op: detects ORB keypoints in `Image<Gray>`.
 *
 * Returns `KeypointSet`. Wraps `cv::ORB`. Defaults: `max_features=500`,
 * `scale_factor=1.2f`, `n_levels=8`.
 *
 * @code
 * KeypointSet ks = gray | DetectORB{}.max_features(200);
 * @endcode
 */
struct DetectORB {
    DetectORB& max_features(int n)   { max_features_ = n; return *this; }
    DetectORB& scale_factor(float f) { scale_factor_ = f; return *this; }
    DetectORB& n_levels(int n)       { n_levels_     = n; return *this; }

    KeypointSet operator()(Image<Gray> img) const;

private:
    int   max_features_{500};
    float scale_factor_{1.2f};
    int   n_levels_{8};
};

/**
 * @brief Pipeline op: detects SIFT keypoints in `Image<Gray>`.
 *
 * Returns `KeypointSet`. Wraps `cv::SIFT` (available since OpenCV 4.4).
 * Defaults: `max_features=0` (no limit), `n_octave_layers=3`.
 *
 * @code
 * KeypointSet ks = gray | DetectSIFT{}.max_features(100);
 * @endcode
 */
struct DetectSIFT {
    DetectSIFT& max_features(int n)    { max_features_    = n; return *this; }
    DetectSIFT& n_octave_layers(int n) { n_octave_layers_ = n; return *this; }

    KeypointSet operator()(Image<Gray> img) const;

private:
    int max_features_{0};    ///< 0 = no limit
    int n_octave_layers_{3};
};

/**
 * @brief Pipeline op: detects AKAZE keypoints in `Image<Gray>`.
 *
 * Returns `KeypointSet`. Wraps `cv::AKAZE`. Default `threshold=0.001f`.
 * Higher threshold → stricter → fewer keypoints.
 *
 * @code
 * KeypointSet ks = gray | DetectAKAZE{}.threshold(0.005f);
 * @endcode
 */
struct DetectAKAZE {
    DetectAKAZE& threshold(float t) { threshold_ = t; return *this; }

    KeypointSet operator()(Image<Gray> img) const;

private:
    float threshold_{0.001f};
};

} // namespace improc::core
