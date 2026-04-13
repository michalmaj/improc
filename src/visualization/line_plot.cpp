// src/visualization/line_plot.cpp
#include "improc/visualization/line_plot.hpp"
#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace improc::visualization {

Image<BGR> LinePlot::operator()(std::vector<float> values) const {
    if (values.empty())
        throw std::invalid_argument("LinePlot: values must not be empty");

    cv::Mat canvas(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));

    // Axis lines
    cv::line(canvas, {0, height_ - 1}, {width_ - 1, height_ - 1}, {80, 80, 80}, 1);
    cv::line(canvas, {0, 0},           {0, height_ - 1},           {80, 80, 80}, 1);

    if (!title_.empty())
        cv::putText(canvas, title_, {4, 16},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, {200, 200, 200}, 1);

    if (values.size() == 1) {
        cv::line(canvas, {0, height_ / 2}, {width_ - 1, height_ / 2}, color_, 1);
        return Image<BGR>(std::move(canvas));
    }

    float v_min = *std::min_element(values.begin(), values.end());
    float v_max = *std::max_element(values.begin(), values.end());
    float range = (v_max > v_min) ? (v_max - v_min) : 1.0f;

    const int margin = 8;
    const int plot_h = height_ - 2 * margin;
    const int plot_w = width_  - 2 * margin;
    const int n      = static_cast<int>(values.size());

    auto to_point = [&](int i) -> cv::Point {
        int x = margin + static_cast<int>(static_cast<float>(i) / (n - 1) * plot_w);
        int y = margin + static_cast<int>((1.0f - (values[i] - v_min) / range) * plot_h);
        return {x, y};
    };

    for (int i = 1; i < n; ++i)
        cv::line(canvas, to_point(i - 1), to_point(i), color_, 1);

    return Image<BGR>(std::move(canvas));
}

} // namespace improc::visualization
