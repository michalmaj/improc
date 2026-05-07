// include/improc/core/ops/draw_matches.hpp
#pragma once
#include <utility>
#include <opencv2/features2d.hpp>
#include "improc/core/ops/matching.hpp"

namespace improc::core {

/**
 * @brief Pipeline op: draws keypoints onto an image.
 *
 * Accepts Image<Gray> or Image<BGR>; always returns Image<BGR>.
 * Uses cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS (oriented, scaled circles).
 * An empty KeypointSet produces a valid BGR image with no annotations.
 *
 * @code
 * Image<BGR> vis = gray | DrawKeypoints{kps};
 * @endcode
 */
struct DrawKeypoints {
    explicit DrawKeypoints(KeypointSet kps) : kps_(std::move(kps)) {}

    Image<BGR> operator()(Image<Gray> img) const;
    Image<BGR> operator()(Image<BGR>  img) const;

private:
    KeypointSet kps_;
};

/**
 * @brief Callable: draws matches between two BGR images side-by-side.
 *
 * Output width: img1.cols + img2.cols. Height: max(img1.rows, img2.rows).
 * An empty MatchSet produces a side-by-side image with no connecting lines.
 *
 * @code
 * Image<BGR> vis = DrawMatches{img1, kps1, img2, kps2, matches}();
 * @endcode
 */
struct DrawMatches {
    DrawMatches(Image<BGR> img1, KeypointSet kps1,
                Image<BGR> img2, KeypointSet kps2,
                MatchSet   ms)
        : img1_(std::move(img1)), kps1_(std::move(kps1)),
          img2_(std::move(img2)), kps2_(std::move(kps2)),
          ms_(std::move(ms)) {}

    Image<BGR> operator()() const;

private:
    Image<BGR>  img1_, img2_;
    KeypointSet kps1_, kps2_;
    MatchSet    ms_;
};

} // namespace improc::core
