// include/improc/core/ops/warp_affine.hpp
#pragma once

#include <optional>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

struct WarpAffine {
    WarpAffine& matrix(const cv::Mat& M) {
        if (M.rows != 2 || M.cols != 3)
            throw ParameterError{"matrix", "must be a 2x3 affine matrix", "WarpAffine"};
        if (M.type() != CV_32F && M.type() != CV_64F)
            throw ParameterError{"matrix", "must be CV_32F or CV_64F", "WarpAffine"};
        M_ = M.clone();
        return *this;
    }
    WarpAffine& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "WarpAffine"};
        width_ = w;
        return *this;
    }
    WarpAffine& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "WarpAffine"};
        height_ = h;
        return *this;
    }

    template<AnyFormat Format>
    Image<Format> operator()(Image<Format> img) const {
        if (!M_)
            throw ParameterError{"matrix", "must be set before calling operator()", "WarpAffine"};
        int w = width_.value_or(img.cols());
        int h = height_.value_or(img.rows());
        cv::Mat dst;
        cv::warpAffine(img.mat(), dst, *M_, cv::Size(w, h));
        return Image<Format>(std::move(dst));
    }

private:
    std::optional<cv::Mat> M_;
    std::optional<int>     width_;
    std::optional<int>     height_;
};

} // namespace improc::core
