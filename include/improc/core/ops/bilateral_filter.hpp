// include/improc/core/ops/bilateral_filter.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Edge-preserving bilateral filter.
 *
 * Smooths pixel values while preserving sharp edges by weighting
 * contributions by both spatial distance and color similarity.
 *
 * @throws improc::ParameterError if diameter, sigma_color, or sigma_space <= 0.
 *
 * @code
 * Image<BGR> smooth = img | BilateralFilter{}.diameter(9).sigma_color(75).sigma_space(75);
 * @endcode
 */
struct BilateralFilter {
    /// @brief Sets filter diameter (pixel neighbourhood size). Must be > 0.
    BilateralFilter& diameter(int d) {
        if (d <= 0)
            throw ParameterError{"diameter", "must be positive", "BilateralFilter"};
        diameter_ = d;
        return *this;
    }
    /// @brief Sets color-space sigma. Larger = wider color range is blurred together.
    BilateralFilter& sigma_color(double s) {
        if (s <= 0.0)
            throw ParameterError{"sigma_color", "must be positive", "BilateralFilter"};
        sigma_color_ = s;
        return *this;
    }
    /// @brief Sets spatial sigma. Larger = farther pixels influence each other.
    BilateralFilter& sigma_space(double s) {
        if (s <= 0.0)
            throw ParameterError{"sigma_space", "must be positive", "BilateralFilter"};
        sigma_space_ = s;
        return *this;
    }

    /// @brief Applies bilateral filter to img.
    Image<Gray> operator()(Image<Gray> img) const;
    /// @brief Applies bilateral filter to img.
    Image<BGR>  operator()(Image<BGR>  img) const;

private:
    int    diameter_    = 9;
    double sigma_color_ = 75.0;
    double sigma_space_ = 75.0;
};

} // namespace improc::core
