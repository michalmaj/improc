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
 * Clipping to [0, 255] is handled by OpenCV's `convertTo` for integer formats.
 * Float formats are not clamped.
 *
 * Works on any `Image<Format>`.
 *
 * @code
 * Image<BGR> bright = img | Brightness{}.delta(50.0);
 * Image<BGR> dark   = img | Brightness{}.delta(-30.0);
 * @endcode
 */
struct Brightness {
    /**
     * @brief Sets the additive brightness offset.
     * @param d Offset applied to every pixel channel; positive brightens, negative darkens.
     * @return Reference to this op for method chaining.
     */
    Brightness& delta(double d) { delta_ = d; return *this; }

    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        cv::Mat dst;
        img.mat().convertTo(dst, -1, 1.0, delta_);
        return Image<F>(std::move(dst));
    }

private:
    double delta_{0.0};
};

} // namespace improc::core
