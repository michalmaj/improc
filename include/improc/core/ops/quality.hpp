/**
 * @file quality.hpp
 * @brief Full-reference image quality metrics (PSNR, SSIM, GMSD, MSE).
 *
 * All ops throw improc::ParameterError when ref.size() != cmp.size().
 */
#pragma once
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Peak Signal-to-Noise Ratio — higher is better.
 *
 * @return INFINITY when images are identical (MSE == 0).
 */
struct PSNR {
    double operator()(const Image<BGR>&  ref, const Image<BGR>&  cmp) const;
    double operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const;
};

/**
 * @brief Structural Similarity Index — higher is better; maximum is 1.0.
 *
 * @return 1.0 for identical images.
 */
struct SSIM {
    double operator()(const Image<BGR>&  ref, const Image<BGR>&  cmp) const;
    double operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const;
};

/**
 * @brief Gradient Magnitude Similarity Deviation — lower is better; 0 for identical images.
 */
struct GMSD {
    double operator()(const Image<BGR>&  ref, const Image<BGR>&  cmp) const;
    double operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const;
};

/**
 * @brief Mean Squared Error — lower is better; 0 for identical images.
 */
struct MSE {
    double operator()(const Image<BGR>&  ref, const Image<BGR>&  cmp) const;
    double operator()(const Image<Gray>& ref, const Image<Gray>& cmp) const;
};

} // namespace improc::core
