// include/improc/core/ops/morphology.hpp
#pragma once

#include <stdexcept>
#include <utility>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::core {

enum class MorphShape { Rect, Cross, Ellipse };

namespace detail {
inline int morph_shape_to_cv(MorphShape s) {
    switch (s) {
        case MorphShape::Rect:    return cv::MORPH_RECT;
        case MorphShape::Cross:   return cv::MORPH_CROSS;
        case MorphShape::Ellipse: return cv::MORPH_ELLIPSE;
    }
    std::unreachable();
}
} // namespace detail

struct Dilate {
    Dilate& kernel_size(int k) {
        if (k <= 0 || k % 2 == 0)
            throw std::invalid_argument("Dilate: kernel_size must be odd and positive");
        kernel_size_ = k; return *this;
    }
    Dilate& iterations(int n) {
        if (n <= 0) throw std::invalid_argument("Dilate: iterations must be positive");
        iterations_ = n; return *this;
    }
    Dilate& shape(MorphShape s) { shape_ = s; return *this; }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat kernel = cv::getStructuringElement(
            detail::morph_shape_to_cv(shape_),
            cv::Size(kernel_size_, kernel_size_));
        cv::Mat dst;
        cv::dilate(img.mat(), dst, kernel, cv::Point(-1, -1), iterations_);
        return Image<Format>(std::move(dst));
    }

private:
    int        kernel_size_ = 3;
    int        iterations_  = 1;
    MorphShape shape_       = MorphShape::Rect;
};

struct Erode {
    Erode& kernel_size(int k) {
        if (k <= 0 || k % 2 == 0)
            throw std::invalid_argument("Erode: kernel_size must be odd and positive");
        kernel_size_ = k; return *this;
    }
    Erode& iterations(int n) {
        if (n <= 0) throw std::invalid_argument("Erode: iterations must be positive");
        iterations_ = n; return *this;
    }
    Erode& shape(MorphShape s) { shape_ = s; return *this; }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        cv::Mat kernel = cv::getStructuringElement(
            detail::morph_shape_to_cv(shape_),
            cv::Size(kernel_size_, kernel_size_));
        cv::Mat dst;
        cv::erode(img.mat(), dst, kernel, cv::Point(-1, -1), iterations_);
        return Image<Format>(std::move(dst));
    }

private:
    int        kernel_size_ = 3;
    int        iterations_  = 1;
    MorphShape shape_       = MorphShape::Rect;
};

} // namespace improc::core
