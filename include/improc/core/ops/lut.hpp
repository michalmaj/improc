// include/improc/core/ops/lut.hpp
#pragma once

#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/// @brief Pipeline op: applies a 256-entry lookup table to every pixel channel.
///
/// `table` must be a 1×256 or 256×1 CV_8UC1 (single-channel) mat.
/// Single-channel tables are applied identically to all image channels.
///
/// @code
/// cv::Mat inv(1, 256, CV_8UC1);
/// for (int i = 0; i < 256; ++i) inv.at<uint8_t>(i) = 255 - i;
/// Image<Gray> result = img | LUT{inv};
/// @endcode
struct LUT {
    explicit LUT(cv::Mat table) : table_(std::move(table)) {
        if (table_.total() != 256)
            throw improc::ParameterError{"table", "must have exactly 256 entries", "LUT"};
        if (table_.depth() != CV_8U)
            throw improc::ParameterError{"table", "must be CV_8U", "LUT"};
    }

    template<AnyFormat F>
    [[nodiscard]] Image<F> operator()(Image<F> img) const {
        cv::Mat result;
        cv::LUT(img.mat(), table_, result);
        return Image<F>(std::move(result));
    }

private:
    cv::Mat table_;
};

} // namespace improc::core
