// include/improc/core/ops/optical_flow.hpp
#pragma once
#include <vector>
#include <opencv2/video/tracking.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Result of sparse Lucas-Kanade optical flow tracking.
 */
struct SparseLKFlowResult {
    std::vector<cv::Point2f> points; ///< Tracked point positions in the next frame.
    std::vector<uchar>       status; ///< 1 if the corresponding point was found, 0 otherwise.
    std::vector<float>       error;  ///< Tracking error for each point.
};

/**
 * @brief Tracks sparse feature points between two grayscale frames using Lucas-Kanade optical flow.
 */
struct SparseLKFlow {
    /// @brief Sets the search window size at each pyramid level (default: {21, 21}).
    SparseLKFlow& win_size(cv::Size s)  { win_size_  = s; return *this; }
    /// @brief Sets the maximum pyramid level (default: 3).
    SparseLKFlow& max_level(int n)      { max_level_ = n; return *this; }
    /// @brief Sets the maximum number of iterations per pyramid level (default: 30).
    SparseLKFlow& max_iter(int n)       { max_iter_  = n; return *this; }
    /// @brief Sets the convergence epsilon for the iterative search (default: 0.01).
    SparseLKFlow& epsilon(double e)     { epsilon_   = e; return *this; }

    SparseLKFlowResult operator()(const Image<Gray>& prev,
                                  const Image<Gray>& next,
                                  const std::vector<cv::Point2f>& prev_pts) const;

private:
    cv::Size win_size_  = {21, 21};
    int      max_level_ = 3;
    int      max_iter_  = 30;
    double   epsilon_   = 0.01;
};

/**
 * @brief Computes dense optical flow using the Farneback polynomial expansion method.
 *
 * @return Image<Flow> (CV_32FC2) where each pixel contains the (dx, dy) displacement vector.
 */
struct DenseFarnebackFlow {
    /// @brief Sets the image scale for pyramid construction (default: 0.5).
    DenseFarnebackFlow& pyr_scale(double s)  { pyr_scale_  = s; return *this; }
    /// @brief Sets the number of pyramid levels (default: 3).
    DenseFarnebackFlow& levels(int n)        { levels_     = n; return *this; }
    /// @brief Sets the averaging window size in pixels (default: 15).
    DenseFarnebackFlow& win_size(int n)      { win_size_   = n; return *this; }
    /// @brief Sets the number of iterations at each pyramid level (default: 3).
    DenseFarnebackFlow& iterations(int n)    { iterations_ = n; return *this; }
    /// @brief Sets the size of the pixel neighbourhood for polynomial expansion (default: 5).
    DenseFarnebackFlow& poly_n(int n)        { poly_n_     = n; return *this; }
    /// @brief Sets the standard deviation for the Gaussian smoothing of polynomial coefficients (default: 1.2).
    DenseFarnebackFlow& poly_sigma(double s) { poly_sigma_ = s; return *this; }

    /// @return Image<Flow> (CV_32FC2) displacement field.
    Image<Flow> operator()(const Image<Gray>& prev, const Image<Gray>& next) const;

private:
    double pyr_scale_  = 0.5;
    int    levels_     = 3;
    int    win_size_   = 15;
    int    iterations_ = 3;
    int    poly_n_     = 5;
    double poly_sigma_ = 1.2;
};

/**
 * @brief Computes dense optical flow using the DIS (Dense Inverse Search) algorithm.
 */
struct DenseDISFlow {
    /// @brief Quality/speed preset for DIS optical flow.
    enum class Preset {
        UltraFast, ///< Lowest quality, fastest computation.
        Fast,      ///< Balanced quality and speed.
        Medium,    ///< Higher quality, slower computation (default).
    };

    /// @brief Sets the quality/speed preset (default: Preset::Medium).
    DenseDISFlow& preset(Preset p) { preset_ = p; return *this; }

    /// @return Image<Flow> (CV_32FC2) displacement field.
    Image<Flow> operator()(const Image<Gray>& prev, const Image<Gray>& next) const;

private:
    Preset preset_ = Preset::Medium;
};

} // namespace improc::core
