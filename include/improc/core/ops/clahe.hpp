// include/improc/core/ops/clahe.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Contrast Limited Adaptive Histogram Equalization (CLAHE).
 *
 * On `Image<Gray>`: CLAHE applied to the single channel directly.
 * On `Image<BGR>`:  converts to LAB color space, applies CLAHE to the L
 *                   channel, then converts back.
 *
 * @throws improc::ParameterError if clip_limit <= 0 or tile dimensions <= 0.
 *
 * @code
 * Image<Gray> sharp = gray | CLAHE{};
 * Image<BGR>  sharp = bgr  | CLAHE{}.clip_limit(2.0).tile_grid_size(8, 8);
 * @endcode
 */
struct CLAHE {
    /// @brief Sets the contrast limit per tile. Higher = more contrast. Default 40.0.
    CLAHE& clip_limit(double limit) {
        if (limit <= 0.0)
            throw ParameterError{"clip_limit", "must be positive", "CLAHE"};
        clip_limit_ = limit;
        return *this;
    }

    /// @brief Sets the tile grid dimensions. Default 8×8.
    CLAHE& tile_grid_size(int w, int h) {
        if (w <= 0) throw ParameterError{"tile_grid_size.w", "must be positive", "CLAHE"};
        if (h <= 0) throw ParameterError{"tile_grid_size.h", "must be positive", "CLAHE"};
        tile_w_ = w;
        tile_h_ = h;
        return *this;
    }

    /// @brief Applies CLAHE to img. May propagate cv::Exception on OpenCV failure.
    Image<Gray> operator()(Image<Gray> img) const;
    /// @brief Applies CLAHE to img. May propagate cv::Exception on OpenCV failure.
    Image<BGR>  operator()(Image<BGR>  img) const;

private:
    double clip_limit_ = 40.0;
    int    tile_w_     = 8;
    int    tile_h_     = 8;
};

} // namespace improc::core
