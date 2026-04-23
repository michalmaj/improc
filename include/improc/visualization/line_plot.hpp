// include/improc/visualization/line_plot.hpp
#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;

/**
 * @brief Renders a 2D line chart from scalar values as `Image<BGR>`.
 *
 * Pass data via `.data()`. X-axis spans 0..N-1; Y-axis auto-scales.
 *
 * @code
 * std::vector<float> vals = {1.0f, 3.5f, 2.0f, 4.1f};
 * Image<BGR> chart = LinePlot{}.width(640).height(360)(vals);
 * @endcode
 */
struct LinePlot {
    /// @brief Sets the chart title text.
    LinePlot& title(std::string t) { title_ = std::move(t); return *this; }
    /// @brief Sets the line color as a BGR scalar.
    LinePlot& color(cv::Scalar c) { color_ = c; return *this; }
    /// @brief Sets the output chart width in pixels.
    /// @throws improc::ParameterError if `w` <= 0.
    LinePlot& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "LinePlot"};
        width_ = w;
        return *this;
    }
    /// @brief Sets the output chart height in pixels.
    /// @throws improc::ParameterError if `h` <= 0.
    LinePlot& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "LinePlot"};
        height_ = h;
        return *this;
    }

    /// @brief Renders the line chart from the given scalar values.
    /// @throws improc::ParameterError if `values` is empty.
    Image<BGR> operator()(const std::vector<float>& values) const;

private:
    std::string title_;
    cv::Scalar  color_  = {255, 255, 255};
    int         width_  = 640;
    int         height_ = 360;
};

} // namespace improc::visualization
