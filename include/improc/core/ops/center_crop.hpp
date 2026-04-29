// include/improc/core/ops/center_crop.hpp
#pragma once

#include <optional>
#include <format>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Crops a centered rectangle from an image.
 *
 * Both `.width()` and `.height()` must be set and must not exceed the
 * source image dimensions. The crop rectangle is centered on the image.
 *
 * @throws improc::ParameterError if width or height is not set.
 * @throws improc::ParameterError if width or height <= 0.
 * @throws improc::ParameterError if the crop exceeds the image dimensions.
 *
 * @code
 * Image<BGR> crop = img | CenterCrop{}.width(224).height(224);
 * @endcode
 */
struct CenterCrop {
    /// @brief Sets crop width in pixels.
    CenterCrop& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "CenterCrop"};
        width_ = w; return *this;
    }
    /// @brief Sets crop height in pixels.
    CenterCrop& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "CenterCrop"};
        height_ = h; return *this;
    }

    /// @brief Applies the centered crop to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!width_ || !height_)
            throw ParameterError{"width/height", "both must be set", "CenterCrop"};
        const int w = *width_, h = *height_;
        if (w > img.cols() || h > img.rows())
            throw ParameterError{"width/height",
                std::format("({},{}) exceeds image size {}x{}",
                    w, h, img.cols(), img.rows()),
                "CenterCrop"};
        const int x = (img.cols() - w) / 2;
        const int y = (img.rows() - h) / 2;
        return Image<Format>(img.mat()(cv::Rect(x, y, w, h)).clone());
    }

private:
    std::optional<int> width_, height_;
};

} // namespace improc::core
