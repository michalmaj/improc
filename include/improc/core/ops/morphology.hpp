// include/improc/core/ops/morphology.hpp
#pragma once

#include <utility>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/// @brief Structuring element shape for morphological operations.
enum class MorphShape {
    Rect,    ///< Rectangular kernel.
    Cross,   ///< Cross-shaped kernel (plus sign).
    Ellipse  ///< Elliptical kernel.
};

namespace detail {
inline int morph_shape_to_cv(MorphShape s) {
    switch (s) {
        case MorphShape::Rect:    return cv::MORPH_RECT;
        case MorphShape::Cross:   return cv::MORPH_CROSS;
        case MorphShape::Ellipse: return cv::MORPH_ELLIPSE;
    }
    std::unreachable();
}
} // namespace detail

/**
 * @brief Morphological dilation — expands bright (foreground) regions.
 *
 * @throws improc::ParameterError if kernel_size is not odd and positive.
 * @throws improc::ParameterError if iterations <= 0.
 *
 * @code
 * Image<BGR> dilated = img | Dilate{}.kernel_size(5).iterations(2);
 * @endcode
 */
struct Dilate {
    /// @brief Sets kernel size. Must be odd and positive.
    Dilate& kernel_size(int k) {
        if (k <= 0 || k % 2 == 0)
            throw ParameterError{"kernel_size",
                std::format("must be odd and positive, got {}", k), "Dilate"};
        kernel_size_ = k; return *this;
    }
    /// @brief Sets number of iterations. Must be >= 1.
    Dilate& iterations(int n) {
        if (n <= 0) throw ParameterError{"iterations", "must be positive", "Dilate"};
        iterations_ = n; return *this;
    }
    /// @brief Sets the structuring element shape. Default: MorphShape::Rect.
    Dilate& shape(MorphShape s) { shape_ = s; return *this; }

    /// @brief Applies morphological dilation to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat kernel = cv::getStructuringElement(
            detail::morph_shape_to_cv(shape_),
            cv::Size(kernel_size_, kernel_size_));
        cv::Mat dst;
        cv::dilate(img.mat(), dst, kernel, cv::Point(-1, -1), iterations_);
        return Image<Format>(std::move(dst));
    }

private:
    int        kernel_size_ = 3;
    int        iterations_  = 1;
    MorphShape shape_       = MorphShape::Rect;
};

/**
 * @brief Morphological erosion — shrinks bright (foreground) regions.
 *
 * @throws improc::ParameterError if kernel_size is not odd and positive.
 * @throws improc::ParameterError if iterations <= 0.
 *
 * @code
 * Image<Gray> eroded = mask | Erode{}.kernel_size(3);
 * @endcode
 */
struct Erode {
    /// @brief Sets kernel size. Must be odd and positive.
    Erode& kernel_size(int k) {
        if (k <= 0 || k % 2 == 0)
            throw ParameterError{"kernel_size",
                std::format("must be odd and positive, got {}", k), "Erode"};
        kernel_size_ = k; return *this;
    }
    /// @brief Sets number of iterations. Must be >= 1.
    Erode& iterations(int n) {
        if (n <= 0) throw ParameterError{"iterations", "must be positive", "Erode"};
        iterations_ = n; return *this;
    }
    /// @brief Sets the structuring element shape. Default: MorphShape::Rect.
    Erode& shape(MorphShape s) { shape_ = s; return *this; }

    /// @brief Applies morphological erosion to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat kernel = cv::getStructuringElement(
            detail::morph_shape_to_cv(shape_),
            cv::Size(kernel_size_, kernel_size_));
        cv::Mat dst;
        cv::erode(img.mat(), dst, kernel, cv::Point(-1, -1), iterations_);
        return Image<Format>(std::move(dst));
    }

private:
    int        kernel_size_ = 3;
    int        iterations_  = 1;
    MorphShape shape_       = MorphShape::Rect;
};

} // namespace improc::core
