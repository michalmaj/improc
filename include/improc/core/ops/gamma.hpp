// include/improc/core/ops/gamma.hpp
#pragma once

#include <opencv2/core.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

// Applies gamma correction: output = input^gamma (per-pixel, per-channel).
//
// gamma < 1  — brightens (e.g. 0.5 = square root)
// gamma = 1  — identity
// gamma > 1  — darkens (e.g. 2.0 = square)
//
// 8-bit images use a precomputed LUT for speed.
// Float images use cv::pow directly.
//
// Works on any Image<Format>.
struct GammaCorrection {
    GammaCorrection& gamma(float g) {
        if (g <= 0.0f)
            throw ParameterError{"gamma", "must be positive", "GammaCorrection"};
        gamma_ = g;
        return *this;
    }

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
