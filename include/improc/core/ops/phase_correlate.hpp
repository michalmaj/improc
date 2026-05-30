// include/improc/core/ops/phase_correlate.hpp
#pragma once
#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Result of phase correlation: sub-pixel translation shift and peak response.
 */
struct PhaseCorrelateResult {
    cv::Point2d shift;    ///< Sub-pixel translation (dx, dy) of next relative to prev.
    double      response; ///< Peak correlation response in [0, 1]; higher means more reliable.
};

/**
 * @brief Estimates the translation between two float grayscale frames using phase correlation.
 *
 * @code
 * auto result = PhaseCorrelate()(prev_f32, next_f32);
 * // result.shift is the sub-pixel (dx, dy) translation
 * @endcode
 */
struct PhaseCorrelate {
    /// @return PhaseCorrelateResult with the sub-pixel shift and peak response strength.
    PhaseCorrelateResult operator()(const Image<Float32>& prev,
                                    const Image<Float32>& next) const;
};

} // namespace improc::core
