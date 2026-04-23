// include/improc/core/ops/threshold.hpp
#pragma once

#include <utility>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/// @brief Thresholding algorithm selector.
enum class ThresholdMode {
    Binary,    ///< `output = pixel > thresh ? max_value : 0`
    BinaryInv, ///< `output = pixel > thresh ? 0 : max_value`
    Truncate,  ///< `output = min(pixel, thresh)`
    ToZero,    ///< `output = pixel > thresh ? pixel : 0`
    ToZeroInv, ///< `output = pixel > thresh ? 0 : pixel`
    Otsu       ///< Binary with automatic Otsu threshold (`.value()` is ignored).
};

namespace detail {
inline int to_cv_type(ThresholdMode m) {
    switch (m) {
        case ThresholdMode::Binary:    return cv::THRESH_BINARY;
        case ThresholdMode::BinaryInv: return cv::THRESH_BINARY_INV;
        case ThresholdMode::Truncate:  return cv::THRESH_TRUNC;
        case ThresholdMode::ToZero:    return cv::THRESH_TOZERO;
        case ThresholdMode::ToZeroInv: return cv::THRESH_TOZERO_INV;
        case ThresholdMode::Otsu:      return cv::THRESH_BINARY | cv::THRESH_OTSU;
    }
    std::unreachable();
}
} // namespace detail

/**
 * @brief Pixel-wise intensity thresholding.
 *
 * Default: Binary mode, threshold value 127, max_value 255.
 * With `ThresholdMode::Otsu`, `.value()` is ignored and the optimal
 * threshold is computed automatically.
 *
 * @code
 * Image<Gray> binary = gray | Threshold{}.value(100).mode(ThresholdMode::Binary);
 * Image<Gray> auto_t = gray | Threshold{}.mode(ThresholdMode::Otsu);
 * @endcode
 *
 * @throws improc::ParameterError if the combination of mode and value is rejected by OpenCV.
 */
struct Threshold {
    /// @brief Sets the threshold value. Ignored when mode is ThresholdMode::Otsu.
    Threshold& value(double v)       { value_     = v; return *this; }
    /// @brief Sets the maximum output value for Binary/BinaryInv modes.
    Threshold& max_value(double v)   { max_value_ = v; return *this; }
    /// @brief Sets the thresholding algorithm. Default: ThresholdMode::Binary.
    Threshold& mode(ThresholdMode m) { mode_      = m; return *this; }

    /// @brief Applies thresholding to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat dst;
        try {
            cv::threshold(img.mat(), dst, value_, max_value_, detail::to_cv_type(mode_));
        } catch (const cv::Exception& e) {
            throw ParameterError{"mode/value",
                std::string(e.what()), "Threshold"};
        }
        return Image<Format>(std::move(dst));
    }

private:
    double        value_     = 127.0;
    double        max_value_ = 255.0;
    ThresholdMode mode_      = ThresholdMode::Binary;
};

} // namespace improc::core
