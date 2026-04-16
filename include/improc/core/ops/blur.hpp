// include/improc/core/ops/blur.hpp
#pragma once

#include <stdexcept>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::core {

struct GaussianBlur {
    GaussianBlur& kernel_size(int k) {
        if (k <= 0 || k % 2 == 0)
            throw std::invalid_argument("GaussianBlur: kernel_size must be odd and positive");
        kernel_size_ = k;
        return *this;
    }
    GaussianBlur& sigma(double s) {
        if (s < 0.0)
            throw std::invalid_argument("GaussianBlur: sigma must be >= 0");
        sigma_ = s;
        return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat dst;
        cv::GaussianBlur(img.mat(), dst, cv::Size(kernel_size_, kernel_size_), sigma_);
        return Image<Format>(std::move(dst));
    }

private:
    int    kernel_size_ = 3;
    double sigma_       = 0.0;
};

struct MedianBlur {
    MedianBlur& kernel_size(int k) {
        if (k <= 0 || k % 2 == 0)
            throw std::invalid_argument("MedianBlur: kernel_size must be odd and positive");
        kernel_size_ = k;
        return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat dst;
        cv::medianBlur(img.mat(), dst, kernel_size_);
        return Image<Format>(std::move(dst));
    }

private:
    int kernel_size_ = 3;
};

} // namespace improc::core
