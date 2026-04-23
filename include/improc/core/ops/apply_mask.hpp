// include/improc/core/ops/apply_mask.hpp
#pragma once

#include <optional>
#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Zeros out pixels where a binary mask is 0.
 *
 * The mask must be a single-channel CV_8UC1 mat the same spatial size
 * as the source image. `.mask()` must be called before `operator()`.
 *
 * @throws improc::ParameterError if mask is not set.
 * @throws improc::ParameterError if mask dimensions differ from the source image.
 *
 * @code
 * Image<BGR> masked = img | ApplyMask{}.mask(binary_mask);
 * @endcode
 */
struct ApplyMask {
    /// @brief Sets the CV_8UC1 binary mask. Must be single-channel; dimension check occurs at call time.
    ApplyMask& mask(Image<Gray> m) {
        if (m.mat().type() != CV_8UC1)
            throw ParameterError{"mask", "must be CV_8UC1", "ApplyMask"};
        mask_ = std::move(m);
        return *this;
    }

    /// @brief Applies the mask to img, zeroing pixels where mask == 0.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!mask_)
            throw ParameterError{"mask", "must be set before calling operator()", "ApplyMask"};
        if (mask_->rows() != img.rows() || mask_->cols() != img.cols())
            throw ParameterError{"mask", "must have the same dimensions as the source image", "ApplyMask"};
        cv::Mat dst = cv::Mat::zeros(img.mat().size(), img.mat().type());
        img.mat().copyTo(dst, mask_->mat());
        return Image<Format>(std::move(dst));
    }

private:
    std::optional<Image<Gray>> mask_;
};

} // namespace improc::core
