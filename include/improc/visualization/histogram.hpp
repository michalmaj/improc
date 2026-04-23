// include/improc/visualization/histogram.hpp
#pragma once

#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;
using improc::core::Gray;
using improc::core::Float32;

/**
 * @brief Renders a pixel-intensity histogram chart as `Image<BGR>`.
 *
 * Supports `Image<BGR>` (three overlapping channel curves),
 * `Image<Gray>` (single gray curve), and `Image<Float32>` (single curve in [0,1]).
 *
 * @code
 * Image<BGR> chart = bgr_img | Histogram{}.bins(256).width(512).height(300);
 * @endcode
 */
struct Histogram {
    /// @brief Sets the number of histogram bins.
    /// @throws improc::ParameterError if `n` <= 0.
    Histogram& bins(int n) {
        if (n <= 0) throw ParameterError{"bins", "must be positive", "Histogram"};
        bins_ = n;
        return *this;
    }
    /// @brief Sets the output chart width in pixels.
    /// @throws improc::ParameterError if `w` <= 0.
    Histogram& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "Histogram"};
        width_ = w;
        return *this;
    }
    /// @brief Sets the output chart height in pixels.
    /// @throws improc::ParameterError if `h` <= 0.
    Histogram& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "Histogram"};
        height_ = h;
        return *this;
    }

    /// @brief Renders a three-channel BGR histogram chart.
    Image<BGR> operator()(Image<BGR>     img) const;
    /// @brief Renders a single-channel grayscale histogram chart.
    Image<BGR> operator()(Image<Gray>    img) const;
    /// @brief Renders a single-channel Float32 histogram chart (values in [0,1]).
    Image<BGR> operator()(Image<Float32> img) const;

private:
    cv::Mat render(const std::vector<cv::Mat>& hists,
                   const std::vector<cv::Scalar>& colors) const;

    int bins_   = 256;
    int width_  = 512;
    int height_ = 256;
};

} // namespace improc::visualization
