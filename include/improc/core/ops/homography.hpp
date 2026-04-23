// include/improc/core/ops/homography.hpp
#pragma once

#include <expected>
#include <optional>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"
#include "improc/error.hpp"

namespace improc::core {

/**
 * @brief Computes a 3×3 projective homography from point correspondences via RANSAC.
 *
 * @param src                Source points (at least 4).
 * @param dst                Corresponding destination points (same count as src).
 * @param ransac_threshold   Max reprojection error in pixels to count as inlier (default 3.0).
 * @return  3×3 CV_64F homography matrix, or an `improc::Error` on failure.
 *
 * @code
 * auto H = find_homography(src_pts, dst_pts);
 * if (!H) { std::cerr << H.error().message; return; }
 * Image<BGR> warped = img | WarpPerspective{}.homography(*H);
 * @endcode
 */
std::expected<cv::Mat, improc::Error>
find_homography(const std::vector<cv::Point2f>& src,
                const std::vector<cv::Point2f>& dst,
                double ransac_threshold = 3.0);

/**
 * @brief Warps an image by a 3×3 perspective homography matrix.
 *
 * `.homography()` must be called before `operator()`. Output size defaults to
 * the source size; use `.width()` / `.height()` for a custom output canvas.
 *
 * @throws improc::ParameterError if homography is not set.
 * @throws improc::ParameterError if the matrix is not 3×3 (CV_32F or CV_64F).
 *
 * @code
 * Image<BGR> warped = img | WarpPerspective{}.homography(H).width(800).height(600);
 * @endcode
 */
struct WarpPerspective {
    WarpPerspective& homography(const cv::Mat& H) {
        if (H.rows != 3 || H.cols != 3)
            throw ParameterError{"homography", "must be a 3x3 matrix", "WarpPerspective"};
        if (H.type() != CV_32F && H.type() != CV_64F)
            throw ParameterError{"homography", "must be CV_32F or CV_64F", "WarpPerspective"};
        H_ = H.clone();
        return *this;
    }
    WarpPerspective& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "WarpPerspective"};
        width_ = w;
        return *this;
    }
    WarpPerspective& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "WarpPerspective"};
        height_ = h;
        return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!H_)
            throw ParameterError{"homography", "must be set before calling operator()", "WarpPerspective"};
        int w = width_.value_or(img.cols());
        int h = height_.value_or(img.rows());
        cv::Mat dst;
        cv::warpPerspective(img.mat(), dst, *H_, cv::Size(w, h));
        return Image<Format>(std::move(dst));
    }

private:
    std::optional<cv::Mat> H_;
    std::optional<int>     width_;
    std::optional<int>     height_;
};

} // namespace improc::core
