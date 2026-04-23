// include/improc/core/ops/gamma.hpp
#pragma once

#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

/**
 * @brief Applies power-law (gamma) correction: `output = input^gamma`.
 *
 * `gamma < 1` brightens; `gamma > 1` darkens; `gamma == 1` is identity.
 * Integer images use a precomputed LUT; float images use `cv::pow`.
 *
 * @throws improc::ParameterError if gamma <= 0.
 *
 * @code
 * Image<BGR> bright = img | GammaCorrection{}.gamma(0.5f);
 * Image<BGR> dark   = img | GammaCorrection{}.gamma(2.0f);
 * @endcode
 */
struct GammaCorrection {
    /// @brief Sets gamma exponent. Must be > 0. Values < 1 brighten; > 1 darken.
    GammaCorrection& gamma(float g) {
        if (g <= 0.0f)
            throw ParameterError{"gamma", "must be positive", "GammaCorrection"};
        gamma_ = g;
        return *this;
    }

    /// @brief Applies gamma correction to img.
    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        const int depth = img.mat().depth();
        cv::Mat dst;

        if (depth == CV_8U) {
            // Build a 256-entry LUT once and apply with cv::LUT
            cv::Mat lut(1, 256, CV_8U);
            auto* p = lut.ptr<uchar>();
            for (int i = 0; i < 256; ++i)
                p[i] = cv::saturate_cast<uchar>(
                    255.0 * std::pow(i / 255.0, static_cast<double>(gamma_)));
            cv::LUT(img.mat(), lut, dst);
        } else {
            // CV_32F: direct power, then clamp to [0, 1]
            cv::pow(img.mat(), static_cast<double>(gamma_), dst);
            cv::min(dst, 1.0, dst);
            cv::max(dst, 0.0, dst);
        }

        return Image<Format>(std::move(dst));
    }

private:
    float gamma_ = 1.0f;
};

} // namespace improc::core
