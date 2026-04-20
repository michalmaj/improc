// include/improc/core/ops/unsharp_mask.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

// Sharpens via blurred subtraction: output = (1+s)*img - s*GaussianBlur(img, sigma).
// Kernel size derived from sigma: ceil(6*sigma) rounded up to next odd integer.
struct UnsharpMask {
    UnsharpMask& sigma(double s) {
        if (s <= 0.0)
            throw ParameterError{"sigma", "must be positive", "UnsharpMask"};
        sigma_ = s;
        return *this;
    }
    UnsharpMask& strength(double s) {
        if (s <= 0.0)
            throw ParameterError{"strength", "must be positive", "UnsharpMask"};
        strength_ = s;
        return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        int k = static_cast<int>(sigma_ * 6);
        if (k % 2 == 0) ++k;
        cv::Mat blurred;
        cv::GaussianBlur(img.mat(), blurred, cv::Size(k, k), sigma_);
        cv::Mat dst;
        cv::addWeighted(img.mat(), 1.0 + strength_, blurred, -strength_, 0.0, dst);
        return Image<Format>(std::move(dst));
    }

private:
    double sigma_    = 1.0;
    double strength_ = 0.5;
};

} // namespace improc::core
