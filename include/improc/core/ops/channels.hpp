// include/improc/core/ops/channels.hpp
#pragma once
#include <array>
#include <stdexcept>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct SplitChannels {
    std::array<Image<Gray>, 3> operator()(const Image<BGR>&  img) const;
    std::array<Image<Gray>, 4> operator()(const Image<BGRA>& img) const;
};

struct MergeChannels {
    Image<BGR>  operator()(const Image<Gray>& b, const Image<Gray>& g,
                            const Image<Gray>& r) const;
    Image<BGRA> operator()(const Image<Gray>& b, const Image<Gray>& g,
                            const Image<Gray>& r, const Image<Gray>& a) const;
};

} // namespace improc::core
