// include/improc/core/ops/clahe.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

// Contrast Limited Adaptive Histogram Equalization (CLAHE).
//
// On Image<Gray>: applied directly to the single channel.
// On Image<BGR>:  converts to LAB, applies CLAHE to the L channel, converts back.
//
// Defaults match OpenCV: clip_limit=40.0, tile_grid=8×8.
//
// Usage:
//   Image<Gray> enhanced = gray | CLAHE{};
//   Image<BGR>  enhanced = bgr  | CLAHE{}.clip_limit(2.0).tile_grid_size(8, 8);
struct CLAHE {
    CLAHE& clip_limit(double limit) {
        if (limit <= 0.0)
            throw ParameterError{"clip_limit", "must be positive", "CLAHE"};
        clip_limit_ = limit;
        return *this;
    }

    CLAHE& tile_grid_size(int w, int h) {
        if (w <= 0) throw ParameterError{"tile_grid_size.w", "must be positive", "CLAHE"};
        if (h <= 0) throw ParameterError{"tile_grid_size.h", "must be positive", "CLAHE"};
        tile_w_ = w;
        tile_h_ = h;
        return *this;
    }

    Image<Gray> operator()(Image<Gray> img) const;
    Image<BGR>  operator()(Image<BGR>  img) const;

private:
    double clip_limit_ = 40.0;
    int    tile_w_     = 8;
    int    tile_h_     = 8;
};

} // namespace improc::core
