// include/improc/io/image_io.hpp
#pragma once

#include <expected>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/format_traits.hpp"
#include "improc/core/concepts.hpp"
#include "improc/error.hpp"

namespace improc::io {

using improc::core::Image;
using improc::core::FormatTraits;
using improc::core::AnyFormat;

namespace detail {

template<AnyFormat F>
cv::Mat convert_raw_to_format(const cv::Mat& raw) {
    // raw is loaded with IMREAD_UNCHANGED — could be Gray (CV_8UC1),
    // BGR (CV_8UC3), or BGRA (CV_8UC4).
    const int target_type = FormatTraits<F>::cv_type;
    if (raw.type() == target_type)
        return raw;

    cv::Mat converted;
    // Try cv::cvtColor if channel counts differ, or convertTo if type differs
    if (raw.channels() != FormatTraits<F>::channels) {
        // Determine the cvtColor code based on raw channels -> target channels
        int code = -1;
        if (raw.channels() == 3 && FormatTraits<F>::channels == 1)
            code = cv::COLOR_BGR2GRAY;
        else if (raw.channels() == 1 && FormatTraits<F>::channels == 3)
            code = cv::COLOR_GRAY2BGR;
        else if (raw.channels() == 4 && FormatTraits<F>::channels == 3)
            code = cv::COLOR_BGRA2BGR;
        else if (raw.channels() == 3 && FormatTraits<F>::channels == 4)
            code = cv::COLOR_BGR2BGRA;
        else if (raw.channels() == 1 && FormatTraits<F>::channels == 4)
            code = cv::COLOR_GRAY2BGRA;

        if (code != -1) {
            cv::cvtColor(raw, converted, code);
        } else {
            converted = raw;
        }
    } else {
        converted = raw;
    }

    // Now handle type conversion (e.g., CV_8UC3 -> CV_32FC3 for Float32C3)
    if (converted.type() != target_type) {
        cv::Mat typed;
        if (FormatTraits<F>::is_float) {
            converted.convertTo(typed, target_type, 1.0 / 255.0);
        } else {
            converted.convertTo(typed, target_type);
        }
        return typed;
    }
    return converted;
}

template<AnyFormat F>
cv::Mat convert_image_to_bgr_or_gray(const Image<F>& img) {
    const cv::Mat& m = img.mat();
    if (m.channels() == 1 || m.channels() == 3) {
        // Already BGR or Gray — just ensure it's CV_8U for writing
        if (m.depth() == CV_8U) return m;
        cv::Mat out;
        m.convertTo(out, (m.channels() == 1) ? CV_8UC1 : CV_8UC3, 255.0);
        return out;
    }
    if (m.channels() == 4) {
        cv::Mat bgr;
        cv::cvtColor(m, bgr, cv::COLOR_BGRA2BGR);
        return bgr;
    }
    return m;
}

} // namespace detail

/**
 * @brief Reads an image file and returns it as `Image<F>`.
 *
 * Loads from @p path using `cv::imread(IMREAD_UNCHANGED)` and converts
 * to the requested format automatically.
 *
 * @tparam F  Target format (BGR, Gray, HSV, Float32C3, …).
 * @return    `Image<F>` on success; `Error` (ImageReadFailed) if the file cannot be read.
 *
 * @code
 * auto result = improc::io::imread<BGR>("photo.png");
 * if (!result) { std::cerr << result.error().message; return; }
 * @endcode
 */
template<AnyFormat F>
std::expected<Image<F>, improc::Error> imread(const std::string& path) {
    cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (raw.empty())
        return std::unexpected(improc::Error::image_read_failed(path));
    cv::Mat converted = detail::convert_raw_to_format<F>(raw);
    return Image<F>(converted);
}

/**
 * @brief Writes an image to a file.
 *
 * Format is inferred from the file extension (.png, .bmp, .tiff, .webp).
 * The image is converted to BGR or Gray (CV_8U) before writing.
 *
 * @return Empty expected on success; `Error` (ImageWriteFailed) on failure.
 *
 * @code
 * auto ok = improc::io::imwrite("output.png", img);
 * if (!ok) { std::cerr << ok.error().message; }
 * @endcode
 */
template<AnyFormat F>
std::expected<void, improc::Error> imwrite(const std::string& path, const Image<F>& img) {
    cv::Mat to_write = detail::convert_image_to_bgr_or_gray(img);
    if (!cv::imwrite(path, to_write))
        return std::unexpected(improc::Error::image_write_failed(path));
    return {};
}

} // namespace improc::io
