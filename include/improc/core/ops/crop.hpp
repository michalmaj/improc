// include/improc/core/ops/crop.hpp
#pragma once

#include <optional>
#include <stdexcept>
#include <format>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct Crop {
    Crop& x(int v)      { x_ = v; return *this; }
    Crop& y(int v)      { y_ = v; return *this; }
    Crop& width(int v)  { width_ = v; return *this; }
    Crop& height(int v) { height_ = v; return *this; }

    template<typename Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!x_ || !y_ || !width_ || !height_) {
            throw std::invalid_argument("Crop: all of x, y, width, height must be set");
        }
        cv::Rect roi(*x_, *y_, *width_, *height_);
        if (roi.width <= 0 || roi.height <= 0 ||
            roi.x < 0 || roi.y < 0 ||
            roi.x + roi.width  > img.cols() ||
            roi.y + roi.height > img.rows()) {
            throw std::invalid_argument(
                std::format("Crop: ROI ({},{},{},{}) out of bounds ({}x{})",
                    roi.x, roi.y, roi.width, roi.height, img.cols(), img.rows()));
        }
        return Image<Format>(img.mat()(roi).clone());
    }

private:
    std::optional<int> x_, y_, width_, height_;
};

} // namespace improc::core
