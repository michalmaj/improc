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

/**
 * @brief Box filter — simple uniform averaging.
 *
 * Each pixel is replaced by the average (or sum if not normalized) of its
 * `kernel_size × kernel_size` neighbourhood.
 *
 * @code
 * Image<BGR> smoothed = img | BoxFilter{}.kernel_size(5).normalize(true);
 * @endcode
 */
struct BoxFilter {
    /// @brief Sets kernel size (width and height). Default is 3.
    BoxFilter& kernel_size(int k) {
        ksize_ = k;
        return *this;
    }

    /// @brief If true (default), result is divided by kernel_size². If false, result is summed.
    BoxFilter& normalize(bool n) {
        normalize_ = n;
        return *this;
    }

    /// @brief Sets border type. Default is cv::BORDER_REFLECT_101.
    BoxFilter& border(int t) {
        border_ = t;
        return *this;
    }

    /// @brief Applies box filter to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat result;
        cv::boxFilter(img.mat(), result, -1,
                      cv::Size(ksize_, ksize_), cv::Point(-1, -1),
                      normalize_, border_);
        return Image<Format>(std::move(result));
    }

private:
    int  ksize_     = 3;
    bool normalize_ = true;
    int  border_    = cv::BORDER_REFLECT_101;
};

} // namespace improc::core
