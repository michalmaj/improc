// src/visualization/scatter.cpp
#include "improc/visualization/scatter.hpp"
#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace improc::visualization {

Image<BGR> Scatter::operator()(const std::vector<float>& xs,
                                const std::vector<float>& ys) const {
    if (xs.empty() || ys.empty())
        throw std::invalid_argument("Scatter: xs and ys must not be empty");
    if (xs.size() != ys.size())
        throw std::invalid_argument("Scatter: xs and ys must have the same size");

    cv::Mat canvas(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));

    // Axis lines
    cv::line(canvas, {0, height_ - 1}, {width_ - 1, height_ - 1}, {80, 80, 80}, 1);
    cv::line(canvas, {0, 0},           {0, height_ - 1},           {80, 80, 80}, 1);

    if (!title_.empty())
        cv::putText(canvas, title_, {4, 16},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, {200, 200, 200}, 1);

    const int margin = 8;
    const int plot_w = width_  - 2 * margin;
    const int plot_h = height_ - 2 * margin;

    auto [x_min_it, x_max_it] = std::minmax_element(xs.begin(), xs.end());
    auto [y_min_it, y_max_it] = std::minmax_element(ys.begin(), ys.end());
    float x_min = *x_min_it, x_max = *x_max_it;
    float y_min = *y_min_it, y_max = *y_max_it;
    float x_range = (x_max > x_min) ? (x_max - x_min) : 1.0f;
    float y_range = (y_max > y_min) ? (y_max - y_min) : 1.0f;

    for (std::size_t i = 0; i < xs.size(); ++i) {
        int px = margin + static_cast<int>((xs[i] - x_min) / x_range * plot_w);
        int py = margin + static_cast<int>((1.0f - (ys[i] - y_min) / y_range) * plot_h);
        cv::circle(canvas, {px, py}, point_radius_, color_, cv::FILLED);
    }

    return Image<BGR>(std::move(canvas));
}

} // namespace improc::visualization
