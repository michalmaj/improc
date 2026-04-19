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

// Compute a projective homography from point correspondences using RANSAC.
// Requires at least 4 matching pairs in src and dst (same size).
// ransac_threshold: max reprojection error in pixels to count as inlier (default 3.0).
std::expected<cv::Mat, improc::Error>
find_homography(const std::vector<cv::Point2f>& src,
                const std::vector<cv::Point2f>& dst,
                double ransac_threshold = 3.0);

// Apply a 3x3 homography matrix to an image via perspective warp.
// Output size: matches input by default; use .width()/.height() for custom size.
// Throws ParameterError if homography() was not called before operator().
struct WarpPerspective {
    WarpPerspective& homography(const cv::Mat& H) {
        H_ = H;
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
