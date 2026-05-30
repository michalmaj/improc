// include/improc/core/ops/channels.hpp
#pragma once
#include <array>
#include <stdexcept>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Splits a multi-channel image into individual grayscale channel images.
 */
struct SplitChannels {
    /// @brief Splits BGR into B, G, R channels.
    /// @return Array of three single-channel Image<Gray> in B, G, R order.
    std::array<Image<Gray>, 3> operator()(const Image<BGR>&  img) const;
    /// @brief Splits BGRA into B, G, R, A channels.
    /// @return Array of four single-channel Image<Gray> in B, G, R, A order.
    std::array<Image<Gray>, 4> operator()(const Image<BGRA>& img) const;
};

/**
 * @brief Merges individual grayscale channel images into a multi-channel image.
 */
struct MergeChannels {
    /// @brief Merges three single-channel images into a BGR image.
    Image<BGR>  operator()(const Image<Gray>& b, const Image<Gray>& g,
                            const Image<Gray>& r) const;
    /// @brief Merges four single-channel images into a BGRA image.
    Image<BGRA> operator()(const Image<Gray>& b, const Image<Gray>& g,
                            const Image<Gray>& r, const Image<Gray>& a) const;
};

} // namespace improc::core
