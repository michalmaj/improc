// include/improc/core/ops/letter_box.hpp
#pragma once

#include <optional>
#include <cmath>
#include <format>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Resizes an image to fit a target canvas while preserving aspect ratio,
 *        padding the remaining area with a constant fill color.
 *
 * Standard "letterbox" transform used in YOLO-style detectors: the image is
 * scaled so the longer dimension matches the target, then padding is added
 * symmetrically on the shorter axis. Output is always exactly `width × height`.
 * Default fill color is `{114, 114, 114}` (YOLO convention).
 *
 * @throws improc::ParameterError if width or height is not set.
 * @throws improc::ParameterError if width or height <= 0.
 *
 * @code
 * // Prepare 640×480 frame for a 640×640 YOLO model
 * Image<BGR> ready = frame | LetterBox{}.width(640).height(640);
 * // Custom fill color
 * Image<BGR> ready2 = frame | LetterBox{}.width(416).height(416).value({0, 0, 0});
 * @endcode
 */
struct LetterBox {
    /// @brief Sets output canvas width in pixels.
    LetterBox& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "LetterBox"};
        width_ = w; return *this;
    }
    /// @brief Sets output canvas height in pixels.
    LetterBox& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "LetterBox"};
        height_ = h; return *this;
    }
    /// @brief Sets the padding fill color (default: {114, 114, 114}).
    LetterBox& value(cv::Scalar v) { value_ = v; return *this; }

    /// @brief Applies the letterbox transform to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!width_ || !height_)
            throw ParameterError{"width/height", "both must be set", "LetterBox"};
        const int tw = *width_, th = *height_;

        const double scale = std::min(
            static_cast<double>(tw) / img.cols(),
            static_cast<double>(th) / img.rows());
        const int new_w = static_cast<int>(std::round(img.cols() * scale));
        const int new_h = static_cast<int>(std::round(img.rows() * scale));

        cv::Mat resized;
        cv::resize(img.mat(), resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

        const int pad_h = th - new_h;
        const int pad_w = tw - new_w;
        const int top    = pad_h / 2;
        const int bottom = pad_h - top;
        const int left   = pad_w / 2;
        const int right  = pad_w - left;

        cv::Mat dst;
        cv::copyMakeBorder(resized, dst,
                           top, bottom, left, right,
                           cv::BORDER_CONSTANT, value_);
        return Image<Format>(std::move(dst));
    }

private:
    std::optional<int> width_, height_;
    cv::Scalar value_ = {114, 114, 114, 0};
};

} // namespace improc::core
