// include/improc/core/ops/edge.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Sobel edge detector — returns gradient magnitude as `Image<Gray>`.
 *
 * Computes Sobel derivatives in X and Y (CV_32F), combines via magnitude,
 * and converts to CV_8U with saturation. BGR input is auto-converted to Gray.
 * `ksize` must be 1, 3, 5, or 7.
 *
 * @throws improc::ParameterError if ksize is not 1, 3, 5, or 7.
 *
 * @code
 * Image<Gray> edges = gray | SobelEdge{}.ksize(3);
 * Image<Gray> edges = bgr  | SobelEdge{};  // BGR auto-converted
 * @endcode
 */
struct SobelEdge {
    /// @brief Sets Sobel kernel size. Must be 1, 3, 5, or 7.
    SobelEdge& ksize(int k) {
        if (k != 1 && k != 3 && k != 5 && k != 7)
            throw ParameterError{"ksize", "must be 1, 3, 5, or 7", "SobelEdge"};
        ksize_ = k;
        return *this;
    }

    /// @brief Detects edges in img using the Sobel operator.
    Image<Gray> operator()(Image<Gray> img) const;
    /// @brief Detects edges in img using the Sobel operator.
    Image<Gray> operator()(Image<BGR>  img) const;

private:
    int ksize_ = 3;
};

/**
 * @brief Canny edge detector — returns binary edge map as `Image<Gray>`.
 *
 * Uses two-threshold hysteresis. A good starting ratio: `threshold2 = 3 * threshold1`.
 * BGR input is auto-converted to Gray.
 *
 * @throws improc::ParameterError if threshold1 or threshold2 < 0.
 * @throws improc::ParameterError if aperture_size is not 3, 5, or 7.
 *
 * @code
 * Image<Gray> edges = img | CannyEdge{}.threshold1(50).threshold2(150);
 * @endcode
 */
struct CannyEdge {
    /// @brief Sets lower hysteresis threshold. Must be >= 0.
    CannyEdge& threshold1(double t) {
        if (t < 0.0)
            throw ParameterError{"threshold1", "must be >= 0", "CannyEdge"};
        threshold1_ = t;
        return *this;
    }
    /// @brief Sets upper hysteresis threshold. Must be >= threshold1.
    CannyEdge& threshold2(double t) {
        if (t < 0.0)
            throw ParameterError{"threshold2", "must be >= 0", "CannyEdge"};
        threshold2_ = t;
        return *this;
    }
    /// @brief Sets Sobel aperture size. Must be 3, 5, or 7.
    CannyEdge& aperture_size(int s) {
        if (s != 3 && s != 5 && s != 7)
            throw ParameterError{"aperture_size", "must be 3, 5, or 7", "CannyEdge"};
        aperture_size_ = s;
        return *this;
    }

    /// @brief Detects edges in img using the Canny algorithm.
    Image<Gray> operator()(Image<Gray> img) const;
    /// @brief Detects edges in img using the Canny algorithm.
    Image<Gray> operator()(Image<BGR>  img) const;

private:
    double threshold1_    = 100.0;
    double threshold2_    = 200.0;
    int    aperture_size_ = 3;
};

} // namespace improc::core
