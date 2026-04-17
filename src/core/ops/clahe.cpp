// src/core/ops/clahe.cpp
#include "improc/core/ops/clahe.hpp"

namespace improc::core {

Image<Gray> CLAHE::operator()(Image<Gray> img) const {
    auto clahe = cv::createCLAHE(clip_limit_, {tile_w_, tile_h_});
    cv::Mat dst;
    clahe->apply(img.mat(), dst);
    return Image<Gray>(std::move(dst));
}

Image<BGR> CLAHE::operator()(Image<BGR> img) const {
    // Convert BGR → LAB (L in [0,255] for CV_8U)
    cv::Mat lab;
    cv::cvtColor(img.mat(), lab, cv::COLOR_BGR2Lab);

    // Split, apply CLAHE to L channel only
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    auto clahe = cv::createCLAHE(clip_limit_, {tile_w_, tile_h_});
    clahe->apply(channels[0], channels[0]);

    cv::merge(channels, lab);

    // Convert LAB → BGR
    cv::Mat dst;
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
    return Image<BGR>(std::move(dst));
}

} // namespace improc::core
