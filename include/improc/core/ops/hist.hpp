// include/improc/core/ops/hist.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

/**
 * @brief Computes a 1-D histogram for a grayscale or BGR image.
 *
 * For BGR images, per-channel histograms are concatenated along columns.
 */
struct CalcHist {
    /// @brief Sets the number of histogram bins (default: 256).
    CalcHist& bins(int b);
    /// @brief Sets the value range [lo, hi) covered by the histogram (default: [0, 256)).
    CalcHist& range(float lo, float hi);

    /// @brief Computes a 1-D histogram for a grayscale image.
    cv::Mat operator()(const Image<Gray>& img) const;
    /// @brief Computes per-channel histograms for a BGR image, concatenated along columns.
    cv::Mat operator()(const Image<BGR>& img) const;

private:
    int bins_ = 256;
    float range_lo_ = 0.0f;
    float range_hi_ = 256.0f;
};

/**
 * @brief Compares two histograms using a configurable comparison method.
 */
struct CompareHist {
    /// @brief Sets the comparison method (default: cv::HISTCMP_CORREL).
    CompareHist& method(int m);

    /// @return Scalar result whose meaning depends on the chosen method.
    double operator()(const cv::Mat& h1, const cv::Mat& h2) const;

private:
    int method_ = cv::HISTCMP_CORREL;
};

} // namespace improc::core
