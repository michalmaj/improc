// include/improc/core/ops/feature_detection.hpp
#pragma once
#include <vector>
#include <cstddef>
#include <utility>
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
 * @warning **Slow at high resolution.** SIFT builds a Gaussian scale-space
 *          pyramid and is significantly slower than ORB.
 *          Measured on Apple M4 Pro (single thread):
 *          - 480×640:   ~14 ms (~71 fps)
 *          - 1080×1920: ~91 ms (~11 fps)
 *          Full detect+describe+match pipeline @ 1080×1920: ~329 ms (~3 fps).
 *          Use `DetectORB` when real-time performance is required
 *          (ORB end-to-end @ 1080×1920: ~94 ms).
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

/**
 * @brief Result of a descriptor-computation operation.
 *
 * Pairs the (possibly pruned) keypoints with their computed descriptors.
 * `cv::Feature2D::compute()` may remove keypoints that cannot be described,
 * so `keypoints` and `descriptors` are always consistent.
 *
 * Descriptor type depends on the algorithm:
 * - ORB / AKAZE: `CV_8U`
 * - SIFT: `CV_32F`
 *
 * @code
 * KeypointSet kps  = gray | DetectORB{};
 * DescriptorSet ds = gray | DescribeORB{kps};
 * // ds.descriptors is CV_8U, ds.size() == ds.keypoints.size()
 * @endcode
 */
struct DescriptorSet {
    KeypointSet keypoints;
    cv::Mat     descriptors;  // CV_32F for SIFT; CV_8U for ORB/AKAZE

    std::size_t size()  const { return keypoints.size(); }
    bool        empty() const { return keypoints.empty(); }
};

/// @brief Pipeline op: computes ORB descriptors for a given `KeypointSet`.
struct DescribeORB {
    explicit DescribeORB(KeypointSet kps) : kps_(std::move(kps)) {}
    DescriptorSet operator()(Image<Gray> img) const;
    DescriptorSet operator()(Image<BGR>  img) const;
private:
    KeypointSet kps_;
};

/**
 * @brief Pipeline op: computes SIFT descriptors for a given `KeypointSet`.
 *
 * @warning **Slow.** SIFT descriptors (128-dimensional float vectors) take
 *          ~9 ms @ 480×640 and ~64 ms @ 1080×1920 on Apple M4 Pro.
 *          Use `DescribeORB` when throughput matters (ORB: ~1 ms @ 480×640).
 */
struct DescribeSIFT {
    explicit DescribeSIFT(KeypointSet kps) : kps_(std::move(kps)) {}
    DescriptorSet operator()(Image<Gray> img) const;
    DescriptorSet operator()(Image<BGR>  img) const;
private:
    KeypointSet kps_;
};

/// @brief Pipeline op: computes AKAZE descriptors for a given `KeypointSet`.
struct DescribeAKAZE {
    explicit DescribeAKAZE(KeypointSet kps) : kps_(std::move(kps)) {}
    DescriptorSet operator()(Image<Gray> img) const;
    DescriptorSet operator()(Image<BGR>  img) const;
private:
    KeypointSet kps_;
};

} // namespace improc::core
