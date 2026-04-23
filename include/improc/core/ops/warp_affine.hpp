// include/improc/core/ops/warp_affine.hpp
#pragma once

#include <optional>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Applies a 2×3 affine transformation matrix to an image.
 *
 * `.matrix()` must be called before `operator()`. Output size defaults to
 * the source image size; use `.width()` / `.height()` for a custom canvas.
 *
 * @throws improc::ParameterError if matrix is not set.
 * @throws improc::ParameterError if the matrix is not 2×3 (CV_32F or CV_64F).
 *
 * @code
 * cv::Mat M = cv::getRotationMatrix2D(center, angle, scale);
 * Image<BGR> warped = img | WarpAffine{}.matrix(M);
 * @endcode
 */
struct WarpAffine {
    /**
     * @brief Sets the 2×3 affine transformation matrix.
     * @throws improc::ParameterError if the matrix is not 2×3 (CV_32F or CV_64F).
     */
    WarpAffine& matrix(const cv::Mat& M) {
        if (M.rows != 2 || M.cols != 3)
            throw ParameterError{"matrix", "must be a 2x3 affine matrix", "WarpAffine"};
        if (M.type() != CV_32F && M.type() != CV_64F)
            throw ParameterError{"matrix", "must be CV_32F or CV_64F", "WarpAffine"};
        M_ = M.clone();
        return *this;
    }
    /// @brief Sets the output canvas width in pixels (default: source width).
    WarpAffine& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "WarpAffine"};
        width_ = w;
        return *this;
    }
    /// @brief Sets the output canvas height in pixels (default: source height).
    WarpAffine& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "WarpAffine"};
        height_ = h;
        return *this;
    }

    /// @brief Applies the affine transform to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!M_)
            throw ParameterError{"matrix", "must be set before calling operator()", "WarpAffine"};
        int w = width_.value_or(img.cols());
        int h = height_.value_or(img.rows());
        cv::Mat dst;
        cv::warpAffine(img.mat(), dst, *M_, cv::Size(w, h));
        return Image<Format>(std::move(dst));
    }

private:
    std::optional<cv::Mat> M_;
    std::optional<int>     width_;
    std::optional<int>     height_;
};

} // namespace improc::core
