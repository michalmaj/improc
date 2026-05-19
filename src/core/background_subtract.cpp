// src/core/background_subtract.cpp
#include "improc/core/ops/background_subtract.hpp"

namespace improc::core {

Image<Gray> BackgroundSubtractMOG2::operator()(const Image<BGR>& img) {
    if (!sub_) {
        sub_ = cv::createBackgroundSubtractorMOG2(history_, threshold_, detect_shadows_);
    }
    cv::Mat fg;
    sub_->apply(img.mat(), fg);
    return Image<Gray>(std::move(fg));
}

Image<Gray> BackgroundSubtractKNN::operator()(const Image<BGR>& img) {
    if (!sub_) {
        sub_ = cv::createBackgroundSubtractorKNN(history_, threshold_, detect_shadows_);
    }
    cv::Mat fg;
    sub_->apply(img.mat(), fg);
    return Image<Gray>(std::move(fg));
}

}  // namespace improc::core
