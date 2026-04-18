// src/core/ops/bilateral_filter.cpp
#include "improc/core/ops/bilateral_filter.hpp"

namespace improc::core {

Image<Gray> BilateralFilter::operator()(Image<Gray> img) const {
    cv::Mat dst;
    cv::bilateralFilter(img.mat(), dst, diameter_, sigma_color_, sigma_space_);
    return Image<Gray>(std::move(dst));
}

Image<BGR> BilateralFilter::operator()(Image<BGR> img) const {
    cv::Mat dst;
    cv::bilateralFilter(img.mat(), dst, diameter_, sigma_color_, sigma_space_);
    return Image<BGR>(std::move(dst));
}

} // namespace improc::core
