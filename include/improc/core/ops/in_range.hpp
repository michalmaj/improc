// include/improc/core/ops/in_range.hpp
#pragma once

#include <optional>
#include <string>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Creates a binary mask by checking each pixel against a per-channel range.
 *
 * Output pixel is 255 when every channel of the source pixel falls within
 * [lower, upper]; 0 otherwise. Always returns `Image<Gray>` (CV_8UC1),
 * regardless of the source format. Both `.lower()` and `.upper()` must be
 * set before calling `operator()`.
 *
 * @throws improc::ParameterError if `lower` or `upper` is not set.
 * @throws improc::ParameterError if the OpenCV call fails.
 *
 * @code
 * // Gray range check
 * Image<Gray> mask = gray | InRange{}.lower({100}).upper({200});
 * // BGR: select green-ish pixels (high G, low B and R)
 * Image<Gray> green = bgr | InRange{}.lower({0, 200, 0}).upper({50, 255, 50});
 * @endcode
 */
struct InRange {
    /// @brief Sets the inclusive lower bound per channel. Default: not set.
    InRange& lower(cv::Scalar v) { lower_ = v; return *this; }
    /// @brief Sets the inclusive upper bound per channel. Default: not set.
    InRange& upper(cv::Scalar v) { upper_ = v; return *this; }

    /// @brief Applies the range check and returns a binary Gray mask.
    template<AnyFormat Format>
    Image<Gray> operator()(Image<Format> img) const {
        if (!lower_)
            throw ParameterError{"lower", "must be set before calling operator()", "InRange"};
        if (!upper_)
            throw ParameterError{"upper", "must be set before calling operator()", "InRange"};
        cv::Mat dst;
        try {
            cv::inRange(img.mat(), *lower_, *upper_, dst);
        } catch (const cv::Exception& e) {
            throw ParameterError{"lower/upper", std::string(e.what()), "InRange"};
        }
        return Image<Gray>(std::move(dst));
    }

private:
    std::optional<cv::Scalar> lower_, upper_;
};

} // namespace improc::core
