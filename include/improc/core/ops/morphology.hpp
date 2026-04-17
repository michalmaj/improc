// include/improc/core/ops/morphology.hpp
#pragma once

#include <utility>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

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
            throw ParameterError{"kernel_size",
                std::format("must be odd and positive, got {}", k), "Dilate"};
        kernel_size_ = k; return *this;
    }
    Dilate& iterations(int n) {
        if (n <= 0) throw ParameterError{"iterations", "must be positive", "Dilate"};
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
            throw ParameterError{"kernel_size",
                std::format("must be odd and positive, got {}", k), "Erode"};
        kernel_size_ = k; return *this;
    }
    Erode& iterations(int n) {
        if (n <= 0) throw ParameterError{"iterations", "must be positive", "Erode"};
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
