// include/improc/visualization/scatter.hpp
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
 * @brief Renders a 2D scatter chart from (x, y) point pairs as `Image<BGR>`.
 *
 * @code
 * std::vector<float> xs = {1.f, 3.f};
 * std::vector<float> ys = {2.f, 1.f};
 * Image<BGR> chart = Scatter{}.operator()(xs, ys);
 * @endcode
 */
struct Scatter {
    /// @brief Sets the chart title text.
    Scatter& title(std::string t) { title_ = std::move(t); return *this; }
    /// @brief Sets the point color as a BGR scalar.
    Scatter& color(cv::Scalar c) { color_ = c; return *this; }
    /// @brief Sets the drawn point radius in pixels.
    /// @throws improc::ParameterError if `r` <= 0.
    Scatter& point_radius(int r) {
        if (r <= 0) throw ParameterError{"point_radius", "must be positive", "Scatter"};
        point_radius_ = r;
        return *this;
    }
    /// @brief Sets the output chart width in pixels.
    /// @throws improc::ParameterError if `w` <= 0.
    Scatter& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "Scatter"};
        width_ = w;
        return *this;
    }
    /// @brief Sets the output chart height in pixels.
    /// @throws improc::ParameterError if `h` <= 0.
    Scatter& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "Scatter"};
        height_ = h;
        return *this;
    }

    /// @brief Renders the scatter chart from parallel x/y value arrays.
    /// @throws improc::ParameterError if `xs` or `ys` is empty, or their sizes differ.
    Image<BGR> operator()(const std::vector<float>& xs,
                          const std::vector<float>& ys) const;

private:
    std::string title_;
    cv::Scalar  color_        = {0, 255, 255};
    int         point_radius_ = 3;
    int         width_        = 512;
    int         height_       = 512;
};

} // namespace improc::visualization
