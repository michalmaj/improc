// include/improc/core/ops/adaptive_threshold.hpp
#pragma once

#include <string>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/// @brief Local neighbourhood method for AdaptiveThreshold.
enum class AdaptiveMethod {
    Mean,     ///< Arithmetic mean of the neighbourhood (cv::ADAPTIVE_THRESH_MEAN_C).
    Gaussian  ///< Gaussian-weighted mean of the neighbourhood (cv::ADAPTIVE_THRESH_GAUSSIAN_C).
};

/**
 * @brief Pixel-wise thresholding with a locally computed threshold.
 *
 * Unlike `Threshold`, the threshold value is computed per-pixel as
 * `local_mean(block_size × block_size neighbourhood) − C`. This makes it
 * robust to non-uniform illumination.
 *
 * Only accepts `Image<Gray>` — `cv::adaptiveThreshold` requires single-channel 8-bit input.
 * Output is always `Image<Gray>` with pixel values 0 or `max_value`.
 *
 * Default: Gaussian method, Binary output, block_size = 11, C = 2.0, max_value = 255.
 *
 * @throws improc::ParameterError if block_size is even or < 3.
 * @throws improc::ParameterError if the OpenCV call fails (e.g. block_size > image dimension).
 *
 * @code
 * // Standard document binarisation
 * Image<Gray> binary = gray | AdaptiveThreshold{}.block_size(11).C(2);
 * // Inverted
 * Image<Gray> inv = gray | AdaptiveThreshold{}.block_size(11).C(2).invert();
 * // Mean method with large neighbourhood
 * Image<Gray> mean_t = gray | AdaptiveThreshold{}.method(AdaptiveMethod::Mean).block_size(31).C(5);
 * @endcode
 */
struct AdaptiveThreshold {
    /// @brief Sets the maximum output value for above-threshold pixels (default: 255).
    AdaptiveThreshold& max_value(double v)     { max_value_  = v; return *this; }
    /// @brief Sets the local neighbourhood method (default: AdaptiveMethod::Gaussian).
    AdaptiveThreshold& method(AdaptiveMethod m) { method_     = m; return *this; }
    /// @brief When called, switches output to BinaryInv (above-threshold → 0).
    AdaptiveThreshold& invert(bool v = true)   { invert_     = v; return *this; }
    /// @brief Sets the neighbourhood block size; must be odd and >= 3 (default: 11).
    AdaptiveThreshold& block_size(int v) {
        if (v < 3 || v % 2 == 0)
            throw ParameterError{"block_size", "must be odd and >= 3", "AdaptiveThreshold"};
        block_size_ = v; return *this;
    }
    /// @brief Sets the constant subtracted from the local mean (default: 2.0).
    AdaptiveThreshold& C(double v) { C_ = v; return *this; }

    /// @brief Applies adaptive thresholding to a grayscale image.
    Image<Gray> operator()(Image<Gray> img) const {
        const int cv_method = (method_ == AdaptiveMethod::Gaussian)
            ? cv::ADAPTIVE_THRESH_GAUSSIAN_C
            : cv::ADAPTIVE_THRESH_MEAN_C;
        const int cv_type = invert_ ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
        cv::Mat dst;
        try {
            cv::adaptiveThreshold(img.mat(), dst,
                                  max_value_, cv_method, cv_type, block_size_, C_);
        } catch (const cv::Exception& e) {
            throw ParameterError{"block_size/C", std::string(e.what()), "AdaptiveThreshold"};
        }
        return Image<Gray>(std::move(dst));
    }

private:
    double         max_value_  = 255.0;
    AdaptiveMethod method_     = AdaptiveMethod::Gaussian;
    bool           invert_     = false;
    int            block_size_ = 11;
    double         C_          = 2.0;
};

} // namespace improc::core
