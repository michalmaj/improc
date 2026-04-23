// include/improc/core/ops/resize.hpp
#pragma once

#include <optional>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Resizes an image to a target width and/or height.
 *
 * If only one dimension is set, the other is computed to preserve the
 * original aspect ratio. Uses bilinear interpolation.
 *
 * @throws improc::ParameterError if neither width nor height is set.
 * @throws improc::ParameterError if width or height <= 0.
 *
 * @code
 * Image<BGR> thumb = img | Resize{}.width(224).height(224);
 * Image<BGR> wide  = img | Resize{}.width(640);  // height auto-computed
 * @endcode
 */
struct Resize {
    /// @brief Sets output width in pixels. Leave unset to auto-compute from height and aspect ratio.
    Resize& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "Resize"};
        width_  = w;
        return *this;
    }
    /// @brief Sets output height in pixels. Leave unset to auto-compute from width and aspect ratio.
    Resize& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "Resize"};
        height_ = h;
        return *this;
    }

    /// @brief Resizes img to the configured dimensions.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!width_ && !height_) {
            throw ParameterError{"width/height", "at least one must be set", "Resize"};
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
