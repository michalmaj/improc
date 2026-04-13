// include/improc/visualization/line_plot.hpp
#pragma once

#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;

struct LinePlot {
    LinePlot& title(std::string t) { title_ = std::move(t); return *this; }
    LinePlot& color(cv::Scalar c) { color_ = c; return *this; }
    LinePlot& width(int w) {
        if (w <= 0) throw std::invalid_argument("LinePlot: width must be positive");
        width_ = w;
        return *this;
    }
    LinePlot& height(int h) {
        if (h <= 0) throw std::invalid_argument("LinePlot: height must be positive");
        height_ = h;
        return *this;
    }

    // Throws std::invalid_argument if values is empty.
    Image<BGR> operator()(const std::vector<float>& values) const;

private:
    std::string title_;
    cv::Scalar  color_  = {255, 255, 255};
    int         width_  = 640;
    int         height_ = 360;
};

} // namespace improc::visualization
