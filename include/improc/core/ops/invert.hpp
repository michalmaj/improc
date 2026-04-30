// include/improc/core/ops/invert.hpp
#pragma once

#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::core {

/**
 * @brief Bitwise inversion of every pixel channel.
 *
 * Each 8-bit channel value v becomes 255 − v. Works on any format;
 * output type matches input. Applying twice restores the original image.
 *
 * @code
 * Image<Gray> inv     = gray | Invert{};
 * Image<BGR>  inv_bgr = bgr  | Invert{};
 * Image<Gray> orig    = gray | Invert{} | Invert{};  // round-trip
 * @endcode
 */
struct Invert {
    /// @brief Applies bitwise NOT to every pixel. Output type matches input.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat dst;
        cv::bitwise_not(img.mat(), dst);
        return Image<Format>(std::move(dst));
    }
};

} // namespace improc::core
