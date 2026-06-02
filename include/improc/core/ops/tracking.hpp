// include/improc/core/ops/tracking.hpp
#pragma once
#include <opencv2/video/tracking.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Return value of `CamShift::operator()`.
 *
 * Holds the orientation-aware bounding box fitted by Continuously Adaptive
 * Mean-Shift. `cv::CamShift` does not expose the iteration count, so only
 * the rotated rectangle is available.
 */
struct CamShiftResult {
    cv::RotatedRect object; ///< Rotated bounding box of the tracked object.
};

/**
 * @brief Continuously Adaptive Mean-Shift (CamShift) tracker.
 *
 * Runs on a probability (back-projection) image and adapts the search
 * window size and orientation each frame. The search window is refined
 * until the centroid shift falls below `epsilon` or `max_iter` is reached.
 * The `window` argument is updated in place with the new axis-aligned
 * bounding box; the returned `CamShiftResult` holds the rotated rectangle.
 *
 * @code
 * CamShift op;
 * op.epsilon(1.0).max_iter(10);
 * cv::Rect window = initial_roi;
 * CamShiftResult r = op(back_proj_image, window);
 * // window is now updated; r.object holds orientation info
 * @endcode
 */
struct CamShift {
    /// @brief Sets the convergence threshold — stops when centroid shift < epsilon (default: 1.0).
    CamShift& epsilon(double e)  { epsilon_  = e; return *this; }
    /// @brief Sets the maximum number of Mean-Shift iterations per frame (default: 10).
    CamShift& max_iter(int n)    { max_iter_ = n; return *this; }

    /// @brief Runs one frame of CamShift on `back_proj`, updating `window` in place.
    [[nodiscard]] CamShiftResult operator()(const Image<Gray>& back_proj, cv::Rect& window) const;

private:
    double epsilon_  = 1.0;
    int    max_iter_ = 10;
};

/**
 * @brief Classic Mean-Shift tracker.
 *
 * Iteratively shifts the search window centroid towards the weighted mean
 * of the probability (back-projection) image until convergence or the
 * iteration limit is reached. The `window` argument is updated in place.
 * Returns the number of iterations actually performed.
 *
 * @code
 * MeanShift op;
 * op.epsilon(1.0).max_iter(10);
 * cv::Rect window = initial_roi;
 * int iters = op(back_proj_image, window);
 * @endcode
 */
struct MeanShift {
    /// @brief Sets the convergence threshold — stops when centroid shift < epsilon (default: 1.0).
    MeanShift& epsilon(double e)  { epsilon_  = e; return *this; }
    /// @brief Sets the maximum number of iterations per frame (default: 10).
    MeanShift& max_iter(int n)    { max_iter_ = n; return *this; }

    /// @brief Runs one frame of Mean-Shift on `back_proj`, updating `window` in place.
    /// @return Number of iterations performed.
    [[nodiscard]] int operator()(const Image<Gray>& back_proj, cv::Rect& window) const;

private:
    double epsilon_  = 1.0;
    int    max_iter_ = 10;
};

} // namespace improc::core
