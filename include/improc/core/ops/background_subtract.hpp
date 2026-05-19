// include/improc/core/ops/background_subtract.hpp
#pragma once

#include <opencv2/video/background_segm.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Foreground/background segmentation using a Gaussian Mixture Model.
 *
 * Stateful — the internal model updates on every `operator()` call. Create
 * once and reuse across frames. Pass as an **lvalue** to `operator|` so the
 * model accumulates across frames; passing a temporary loses the state.
 *
 * The model is created lazily on the first `operator()` call using the
 * parameters set by the fluent setters. Setters called after the first
 * `operator()` call have no effect.
 *
 * Output: `Image<Gray>` foreground mask (255 = foreground, 127 = shadow if
 * `detect_shadows` is true, 0 = background).
 *
 * @code
 * BackgroundSubtractMOG2 sub;
 * sub.history(500).threshold(16.0).detect_shadows(true);
 * // In a frame loop:
 * Image<Gray> fg = *frame.rgb | sub;
 * @endcode
 */
struct BackgroundSubtractMOG2 {
    BackgroundSubtractMOG2() = default;

    /// @brief Number of frames used to model the background. Default: 500.
    BackgroundSubtractMOG2& history(int h)         { history_ = h;         return *this; }
    /// @brief Mahalanobis distance threshold for classifying foreground. Default: 16.0.
    BackgroundSubtractMOG2& threshold(double t)    { threshold_ = t;       return *this; }
    /// @brief Whether to detect and label shadows (127) separately. Default: true.
    BackgroundSubtractMOG2& detect_shadows(bool s) { detect_shadows_ = s;  return *this; }

    /// @brief Applies the subtractor; updates internal model. Non-const.
    Image<Gray> operator()(const Image<BGR>& img);

private:
    int    history_        = 500;
    double threshold_      = 16.0;
    bool   detect_shadows_ = true;
    cv::Ptr<cv::BackgroundSubtractorMOG2> sub_;
};

/**
 * @brief Foreground/background segmentation using K-Nearest Neighbours.
 *
 * Same stateful semantics as `BackgroundSubtractMOG2`. Faster than MOG2 in
 * controlled environments with stable illumination.
 *
 * @code
 * BackgroundSubtractKNN sub;
 * sub.history(500).threshold(400.0).detect_shadows(false);
 * // In a frame loop:
 * Image<Gray> fg = *frame.rgb | sub;
 * @endcode
 */
struct BackgroundSubtractKNN {
    BackgroundSubtractKNN() = default;

    /// @brief Number of frames used to model the background. Default: 500.
    BackgroundSubtractKNN& history(int h)         { history_ = h;         return *this; }
    /// @brief Squared distance threshold for classifying foreground. Default: 400.0.
    BackgroundSubtractKNN& threshold(double t)    { threshold_ = t;       return *this; }
    /// @brief Whether to detect and label shadows (127) separately. Default: true.
    BackgroundSubtractKNN& detect_shadows(bool s) { detect_shadows_ = s;  return *this; }

    /// @brief Applies the subtractor; updates internal model. Non-const.
    Image<Gray> operator()(const Image<BGR>& img);

private:
    int    history_        = 500;
    double threshold_      = 400.0;
    bool   detect_shadows_ = true;
    cv::Ptr<cv::BackgroundSubtractorKNN> sub_;
};

}  // namespace improc::core
