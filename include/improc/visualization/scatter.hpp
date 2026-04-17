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

struct Scatter {
    Scatter& title(std::string t) { title_ = std::move(t); return *this; }
    Scatter& color(cv::Scalar c) { color_ = c; return *this; }
    Scatter& point_radius(int r) {
        if (r <= 0) throw ParameterError{"point_radius", "must be positive", "Scatter"};
        point_radius_ = r;
        return *this;
    }
    Scatter& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "Scatter"};
        width_ = w;
        return *this;
    }
    Scatter& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "Scatter"};
        height_ = h;
        return *this;
    }

    // Throws ParameterError if xs or ys is empty, or xs.size() != ys.size().
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
