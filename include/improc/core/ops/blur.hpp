// include/improc/core/ops/blur.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Gaussian blur — frequency-domain smoothing.
 *
 * Applies a Gaussian kernel of size `kernel_size × kernel_size`.
 * `sigma = 0` lets OpenCV compute sigma from the kernel size.
 *
 * @throws improc::ParameterError if kernel_size is not odd and positive.
 * @throws improc::ParameterError if sigma < 0.
 *
 * @code
 * Image<BGR> blurred = img | GaussianBlur{}.kernel_size(5).sigma(1.5);
 * @endcode
 */
struct GaussianBlur {
    /// @brief Sets kernel size. Must be odd and positive.
    GaussianBlur& kernel_size(int k) {
        if (k <= 0 || k % 2 == 0)
            throw ParameterError{"kernel_size",
                std::format("must be odd and positive, got {}", k), "GaussianBlur"};
        kernel_size_ = k;
        return *this;
    }
    /// @brief Sets Gaussian sigma. 0 = auto-computed from kernel size.
    GaussianBlur& sigma(double s) {
        if (s < 0.0)
            throw ParameterError{"sigma", "must be >= 0", "GaussianBlur"};
        sigma_ = s;
        return *this;
    }

    /// @brief Applies Gaussian blur to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat dst;
        cv::GaussianBlur(img.mat(), dst, cv::Size(kernel_size_, kernel_size_), sigma_);
        return Image<Format>(std::move(dst));
    }

private:
    int    kernel_size_ = 3;
    double sigma_       = 0.0;
};

/**
 * @brief Median blur — impulse noise removal.
 *
 * Replaces each pixel with the median of its `kernel_size × kernel_size`
 * neighbourhood. Effective for salt-and-pepper noise.
 *
 * @throws improc::ParameterError if kernel_size is not odd and positive.
 *
 * @code
 * Image<BGR> clean = img | MedianBlur{}.kernel_size(5);
 * @endcode
 */
struct MedianBlur {
    /// @brief Sets kernel size. Must be odd and positive.
    MedianBlur& kernel_size(int k) {
        if (k <= 0 || k % 2 == 0)
            throw ParameterError{"kernel_size",
                std::format("must be odd and positive, got {}", k), "MedianBlur"};
        kernel_size_ = k;
        return *this;
    }

    /// @brief Applies median blur to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat dst;
        cv::medianBlur(img.mat(), dst, kernel_size_);
        return Image<Format>(std::move(dst));
    }

private:
    int kernel_size_ = 3;
};

} // namespace improc::core
