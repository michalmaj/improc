// include/improc/core/ops/pad.hpp
#pragma once

#include <utility>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/// @brief Border fill strategy for padding operations.
enum class PadMode {
    Constant,  ///< Fill with a constant color (default: black).
    Reflect,   ///< Mirror-reflect border pixels.
    Replicate  ///< Repeat the edge pixel.
};

namespace detail {
inline int pad_mode_to_cv(PadMode m) {
    switch (m) {
        case PadMode::Constant:  return cv::BORDER_CONSTANT;
        case PadMode::Reflect:   return cv::BORDER_REFLECT;
        case PadMode::Replicate: return cv::BORDER_REPLICATE;
    }
    std::unreachable();
}
} // namespace detail

/**
 * @brief Adds border pixels to one or more sides of an image.
 *
 * Border style is controlled by `.mode()`:
 * `PadMode::Constant` fills with `.value()` (default: black),
 * `PadMode::Reflect` mirrors border pixels,
 * `PadMode::Replicate` repeats the edge pixel.
 *
 * @throws improc::ParameterError if all padding sides are 0.
 *
 * @code
 * Image<BGR> padded = img | Pad{}.top(10).bottom(10).left(5).right(5);
 * @endcode
 */
struct Pad {
    /// @brief Sets top border width in pixels.
    Pad& top(int v) {
        if (v < 0) throw ParameterError{"top", "must be >= 0", "Pad"};
        top_ = v; return *this;
    }
    /// @brief Sets bottom border width in pixels.
    Pad& bottom(int v) {
        if (v < 0) throw ParameterError{"bottom", "must be >= 0", "Pad"};
        bottom_ = v; return *this;
    }
    /// @brief Sets left border width in pixels.
    Pad& left(int v) {
        if (v < 0) throw ParameterError{"left", "must be >= 0", "Pad"};
        left_ = v; return *this;
    }
    /// @brief Sets right border width in pixels.
    Pad& right(int v) {
        if (v < 0) throw ParameterError{"right", "must be >= 0", "Pad"};
        right_ = v; return *this;
    }
    /// @brief Sets the border fill strategy (default: PadMode::Constant).
    Pad& mode(PadMode m)      { mode_  = m; return *this; }
    /// @brief Sets the fill color used when mode is PadMode::Constant.
    Pad& value(cv::Scalar v)  { value_ = v; return *this; }

    /// @brief Applies the configured border padding to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (top_ == 0 && bottom_ == 0 && left_ == 0 && right_ == 0)
            throw ParameterError{"top/bottom/left/right", "at least one side must be > 0", "Pad"};
        cv::Mat dst;
        try {
            cv::copyMakeBorder(img.mat(), dst,
                               top_, bottom_, left_, right_,
                               detail::pad_mode_to_cv(mode_), value_);
        } catch (const cv::Exception& e) {
            throw ParameterError{"mode", std::string(e.what()), "Pad"};
        }
        return Image<Format>(std::move(dst));
    }

private:
    int        top_ = 0, bottom_ = 0, left_ = 0, right_ = 0;
    PadMode    mode_  = PadMode::Constant;
    cv::Scalar value_ = {0, 0, 0, 0};
};

/**
 * @brief Pads the shorter dimension to produce a square image.
 *
 * Adds equal padding to both sides of the short axis. Border style
 * is controlled by `.mode()` (default: `PadMode::Constant`, black).
 *
 * @throws improc::ParameterError if the OpenCV border operation fails.
 *
 * @code
 * Image<BGR> square = img | PadToSquare{};
 * @endcode
 */
struct PadToSquare {
    /// @brief Sets the border fill strategy (default: PadMode::Constant).
    PadToSquare& mode(PadMode m)     { mode_  = m; return *this; }
    /// @brief Sets the fill color used when mode is PadMode::Constant.
    PadToSquare& value(cv::Scalar v) { value_ = v; return *this; }

    /// @brief Pads img along the short axis to produce a square.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        const int h = img.rows();
        const int w = img.cols();
        if (h == w) return img.clone();

        const int diff   = std::abs(h - w);
        const int pad1   = diff / 2;
        const int pad2   = diff - pad1;
        const int top    = (h < w) ? pad1 : 0;
        const int bottom = (h < w) ? pad2 : 0;
        const int left   = (w < h) ? pad1 : 0;
        const int right  = (w < h) ? pad2 : 0;

        cv::Mat dst;
        try {
            cv::copyMakeBorder(img.mat(), dst,
                               top, bottom, left, right,
                               detail::pad_mode_to_cv(mode_), value_);
        } catch (const cv::Exception& e) {
            throw ParameterError{"mode", std::string(e.what()), "PadToSquare"};
        }
        return Image<Format>(std::move(dst));
    }

private:
    PadMode    mode_  = PadMode::Constant;
    cv::Scalar value_ = {0, 0, 0, 0};
};

} // namespace improc::core
