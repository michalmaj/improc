// include/improc/core/ops/contrast.hpp
#pragma once

#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Multiplicative contrast adjustment.
 *
 * Applies `output = clip(input * factor)` to every pixel channel.
 * `factor > 1` increases contrast; `0 < factor < 1` decreases contrast.
 * Clipping to [0, 255] is handled by OpenCV's `convertTo` for integer formats.
 * Float formats are not clamped.
 *
 * @throws ParameterError if factor <= 0.
 *
 * @code
 * Image<BGR> more = img | Contrast{}.factor(1.5);
 * Image<BGR> less = img | Contrast{}.factor(0.7);
 * @endcode
 */
struct Contrast {
    Contrast& factor(double f) {
        if (f <= 0.0)
            throw ParameterError{"factor", "must be > 0", "Contrast"};
        factor_ = f;
        return *this;
    }

    template<AnyFormat F>
    Image<F> operator()(Image<F> img) const {
        cv::Mat dst;
        img.mat().convertTo(dst, -1, factor_, 0.0);
        return Image<F>(std::move(dst));
    }

private:
    double factor_{1.0};
};

} // namespace improc::core
