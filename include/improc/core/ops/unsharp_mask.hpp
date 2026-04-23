// include/improc/core/ops/unsharp_mask.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Sharpens via unsharp masking: `output = (1+s)*img − s*GaussianBlur(img, σ)`.
 *
 * Larger `strength` amplifies edges more aggressively;
 * larger `sigma` widens the blur kernel used for the mask.
 *
 * @throws improc::ParameterError if sigma or strength <= 0.
 *
 * @code
 * Image<BGR> sharp = img | UnsharpMask{}.sigma(1.0).strength(0.5);
 * @endcode
 */
struct UnsharpMask {
    /// @brief Sets blur kernel sigma for the mask. Must be > 0.
    UnsharpMask& sigma(double s) {
        if (s <= 0.0)
            throw ParameterError{"sigma", "must be positive", "UnsharpMask"};
        sigma_ = s;
        return *this;
    }
    /// @brief Sets sharpening strength. Must be > 0. Larger = stronger sharpening.
    UnsharpMask& strength(double s) {
        if (s <= 0.0)
            throw ParameterError{"strength", "must be positive", "UnsharpMask"};
        strength_ = s;
        return *this;
    }

    /// @brief Applies unsharp mask sharpening to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat blurred;
        cv::GaussianBlur(img.mat(), blurred, cv::Size(0, 0), sigma_);
        cv::Mat dst;
        cv::addWeighted(img.mat(), 1.0 + strength_, blurred, -strength_, 0.0, dst);
        return Image<Format>(std::move(dst));
    }

private:
    double sigma_    = 1.0;
    double strength_ = 0.5;
};

} // namespace improc::core
