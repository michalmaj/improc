// include/improc/core/ops/brightness.hpp
#pragma once

#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::core {

/**
 * @brief Additive brightness adjustment.
 *
 * Applies `output = clip(input + delta)` to every pixel channel.
 * Positive delta brightens; negative delta darkens.
 * Clipping is handled by OpenCV's `convertTo`.
 *
 * Works on any `Image<Format>`.
 *
 * @code
 * Image<BGR> bright = img | Brightness{}.delta(50.0);
 * Image<BGR> dark   = img | Brightness{}.delta(-30.0);
 * @endcode
 */
struct Brightness {
    Brightness& delta(double d) { delta_ = d; return *this; }

    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        cv::Mat dst;
        img.mat().convertTo(dst, -1, 1.0, delta_);
        return Image<F>(dst);
    }

private:
    double delta_{0.0};
};

} // namespace improc::core
