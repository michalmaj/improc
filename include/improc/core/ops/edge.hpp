// include/improc/core/ops/edge.hpp
#pragma once

#include <format>
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
    /// @brief Sets upper hysteresis threshold. Must be >= 0.
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

/**
 * @brief Laplacian edge detector — returns second-derivative edge map as `Image<Gray>`.
 *
 * Computes `cv::Laplacian` at CV_16S depth (to capture negative responses),
 * then folds via `cv::convertScaleAbs` to CV_8U. BGR input is auto-converted to Gray.
 * `ksize` must be odd and positive. `scale` must be > 0. `delta` accepts any value.
 *
 * @throws improc::ParameterError if ksize is not odd and positive.
 * @throws improc::ParameterError if scale is not positive.
 *
 * @code
 * Image<Gray> edges = gray | LaplacianEdge{};
 * Image<Gray> edges = bgr  | LaplacianEdge{}.ksize(3).scale(2.0).delta(128.0);
 * @endcode
 */
struct LaplacianEdge {
    /// @brief Laplacian kernel size. Default 1. Must be odd and positive.
    LaplacianEdge& ksize(int v) {
        if (v <= 0 || v % 2 == 0)
            throw ParameterError{"ksize",
                std::format("must be odd and positive, got {}", v), "LaplacianEdge"};
        ksize_ = v;
        return *this;
    }
    /// @brief Scale factor applied to computed Laplacian values. Default 1.0. Must be > 0.
    LaplacianEdge& scale(double v) {
        if (v <= 0.0)
            throw ParameterError{"scale", "must be positive", "LaplacianEdge"};
        scale_ = v;
        return *this;
    }
    /// @brief Offset added to Laplacian values before output. Default 0.0. Any value.
    LaplacianEdge& delta(double v) {
        delta_ = v;
        return *this;
    }

    /// @brief Detects edges in img using the Laplacian operator.
    Image<Gray> operator()(Image<Gray> img) const;
    /// @brief Detects edges in img using the Laplacian operator (auto-converts BGR→Gray).
    Image<Gray> operator()(Image<BGR>  img) const;

private:
    int    ksize_ = 1;
    double scale_ = 1.0;
    double delta_ = 0.0;
};

/**
 * @brief Harris corner detector — returns corner response map as `Image<Gray>`.
 *
 * Computes `cv::cornerHarris`, normalizes to [0, 255] via `cv::NORM_MINMAX`,
 * and converts to CV_8U. Brighter pixels = stronger corner response.
 * BGR input is auto-converted to Gray. Typical k range: 0.04–0.06.
 *
 * @throws improc::ParameterError if block_size <= 0.
 * @throws improc::ParameterError if ksize is not 3, 5, or 7.
 * @throws improc::ParameterError if k is not in (0, 1).
 *
 * @code
 * Image<Gray> corners = gray | HarrisCorner{};
 * Image<Gray> c2      = bgr  | HarrisCorner{}.block_size(3).ksize(5).k(0.05);
 * @endcode
 */
struct HarrisCorner {
    /// @brief Neighbourhood size for covariance matrix. Default 2. Must be > 0.
    HarrisCorner& block_size(int v) {
        if (v <= 0)
            throw ParameterError{"block_size", "must be positive", "HarrisCorner"};
        block_size_ = v;
        return *this;
    }
    /// @brief Sobel kernel size. Must be 3, 5, or 7. Default 3.
    HarrisCorner& ksize(int v) {
        if (v != 3 && v != 5 && v != 7)
            throw ParameterError{"ksize", "must be 3, 5, or 7", "HarrisCorner"};
        ksize_ = v;
        return *this;
    }
    /// @brief Harris sensitivity parameter. Must be in (0, 1). Default 0.04.
    HarrisCorner& k(double v) {
        if (v <= 0.0 || v >= 1.0)
            throw ParameterError{"k", "must be in (0, 1)", "HarrisCorner"};
        k_ = v;
        return *this;
    }

    /// @brief Returns normalized corner response map for a gray image.
    Image<Gray> operator()(Image<Gray> img) const;
    /// @brief Returns normalized corner response map (auto-converts BGR→Gray).
    Image<Gray> operator()(Image<BGR>  img) const;

private:
    int    block_size_ = 2;
    int    ksize_      = 3;
    double k_          = 0.04;
};

} // namespace improc::core
