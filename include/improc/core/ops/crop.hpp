// include/improc/core/ops/crop.hpp
#pragma once

#include <optional>
#include <format>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

struct Crop {
    Crop& x(int v)      { x_ = v; return *this; }
    Crop& y(int v)      { y_ = v; return *this; }
    Crop& width(int v) {
        if (v <= 0) throw ParameterError{"width", "must be positive", "Crop"};
        width_ = v; return *this;
    }
    Crop& height(int v) {
        if (v <= 0) throw ParameterError{"height", "must be positive", "Crop"};
        height_ = v; return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!x_ || !y_ || !width_ || !height_)
            throw ParameterError{"x/y/width/height", "all four must be set", "Crop"};
        cv::Rect roi(*x_, *y_, *width_, *height_);
        if (roi.x < 0 || roi.y < 0 ||
            roi.x + roi.width  > img.cols() ||
            roi.y + roi.height > img.rows())
            throw ParameterError{"ROI",
                std::format("({},{},{},{}) out of bounds for image {}x{}",
                    roi.x, roi.y, roi.width, roi.height, img.cols(), img.rows()),
                "Crop"};
        return Image<Format>(img.mat()(roi).clone());
    }

private:
    std::optional<int> x_, y_, width_, height_;
};

} // namespace improc::core
