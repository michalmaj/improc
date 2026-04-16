// include/improc/core/ops/resize.hpp
#pragma once

#include <optional>
#include <stdexcept>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::core {

struct Resize {
    Resize& width(int w) {
        if (w <= 0) throw std::invalid_argument("Resize: width must be positive");
        width_  = w;
        return *this;
    }
    Resize& height(int h) {
        if (h <= 0) throw std::invalid_argument("Resize: height must be positive");
        height_ = h;
        return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!width_ && !height_) {
            throw std::invalid_argument("Resize: at least one of width or height must be set");
        }

        int w = width_.value_or(0);
        int h = height_.value_or(0);
        // img is guaranteed non-empty by Image constructor, so rows/cols > 0
        if (!width_) {
            w = static_cast<int>(std::round(static_cast<double>(img.cols()) * h / img.rows()));
        } else if (!height_) {
            h = static_cast<int>(std::round(static_cast<double>(img.rows()) * w / img.cols()));
        }
        cv::Mat dst;
        cv::resize(img.mat(), dst, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        return Image<Format>(std::move(dst));
    }

private:
    std::optional<int> width_;
    std::optional<int> height_;
};

} // namespace improc::core
