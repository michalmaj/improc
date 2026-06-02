// include/improc/core/ops/grabcut.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Segments the foreground from the background using the GrabCut algorithm.
 *
 * @code
 * auto mask = GrabCut().iterations(5)(img, cv::Rect{50, 50, 200, 300});
 * @endcode
 */
struct GrabCut {
    /// @brief Sets the number of EM iterations (default: 5).
    GrabCut& iterations(int n);

    /// @brief Runs GrabCut and returns a binary foreground mask.
    /// @param roi Rectangle containing the foreground object.
    /// @return Image<Gray> where 255 = foreground, 0 = background.
    [[nodiscard]] Image<Gray> operator()(const Image<BGR>& img, cv::Rect roi) const;

private:
    int iterations_{5};
};

} // namespace improc::core
