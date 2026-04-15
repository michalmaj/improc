// include/improc/core/ops/threshold.hpp
#pragma once

#include <stdexcept>
#include <utility>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

enum class ThresholdMode { Binary, BinaryInv, Truncate, ToZero, ToZeroInv, Otsu };

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

struct Threshold {
    Threshold& value(double v)       { value_     = v; return *this; }
    Threshold& max_value(double v)   { max_value_ = v; return *this; }
    Threshold& mode(ThresholdMode m) { mode_      = m; return *this; }

    template<typename Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat dst;
        try {
            cv::threshold(img.mat(), dst, value_, max_value_, detail::to_cv_type(mode_));
        } catch (const cv::Exception& e) {
            throw std::runtime_error("Threshold: " + std::string(e.what()));
        }
        return Image<Format>(std::move(dst));
    }

private:
    double        value_     = 127.0;
    double        max_value_ = 255.0;
    ThresholdMode mode_      = ThresholdMode::Binary;
};

} // namespace improc::core
