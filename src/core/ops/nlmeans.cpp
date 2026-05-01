// src/core/ops/nlmeans.cpp
#include "improc/core/ops/nlmeans.hpp"

namespace improc::core {

Image<Gray> NLMeansDenoising::operator()(Image<Gray> img) const {
    cv::Mat dst;
    cv::fastNlMeansDenoising(img.mat(), dst, h_,
                              template_window_size_, search_window_size_);
    return Image<Gray>(std::move(dst));
}

Image<BGR> NLMeansDenoising::operator()(Image<BGR> img) const {
    cv::Mat dst;
    cv::fastNlMeansDenoisingColored(img.mat(), dst, h_, h_color_,
                                     template_window_size_, search_window_size_);
    return Image<BGR>(std::move(dst));
}

} // namespace improc::core
