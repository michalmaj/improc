// include/improc/core/ops/invert.hpp
#pragma once

#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::core {

/**
 * @brief Bitwise inversion of every pixel channel.
 *
 * Each channel value v becomes its bitwise complement. For 8-bit formats
 * (Gray, BGR, BGRA), this maps v → 255 − v. Only integer formats are
 * supported (not Float32 / Float32C3). Applying twice restores the original.
 *
 * @code
 * Image<Gray> inv     = gray | Invert{};
 * Image<BGR>  inv_bgr = bgr  | Invert{};
 * Image<Gray> orig    = gray | Invert{} | Invert{};  // round-trip
 * @endcode
 */
struct Invert {
    template<IntegerFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat dst;
        cv::bitwise_not(img.mat(), dst);
        return Image<Format>(std::move(dst));
    }
};

} // namespace improc::core
