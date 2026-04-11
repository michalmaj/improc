// src/core/ops/normalize.cpp
#include "improc/core/ops/normalize.hpp"
#include <cmath>
#include <opencv2/core.hpp>

namespace improc::core {

Image<Float32> Normalize::operator()(Image<Float32> img) const {
    double min_val, max_val;
    cv::minMaxLoc(img.mat(), &min_val, &max_val);
    cv::Mat dst;
    if (std::abs(max_val - min_val) < 1e-10) {
        dst = cv::Mat::zeros(img.mat().size(), CV_32FC1);
    } else {
        const double scale = 1.0 / (max_val - min_val);
        const double shift = -min_val * scale;
        img.mat().convertTo(dst, CV_32FC1, scale, shift);
    }
    return Image<Float32>(std::move(dst));
}

NormalizeTo::NormalizeTo(float min, float max) : min_(min), max_(max) {
    if (min >= max) {
        throw std::invalid_argument("NormalizeTo: min must be less than max");
    }
}

Image<Float32> NormalizeTo::operator()(Image<Float32> img) const {
    double min_val, max_val;
    cv::minMaxLoc(img.mat(), &min_val, &max_val);
    cv::Mat dst;
    if (std::abs(max_val - min_val) < 1e-10) {
        dst = cv::Mat::zeros(img.mat().size(), CV_32FC1);
    } else {
        const double scale = (max_ - min_) / (max_val - min_val);
        const double shift = min_ - min_val * scale;
        img.mat().convertTo(dst, CV_32FC1, scale, shift);
    }
    return Image<Float32>(std::move(dst));
}

Standardize::Standardize(float mean, float std_dev) : mean_(mean), std_dev_(std_dev) {
    if (std_dev <= 0.0f) {
        throw std::invalid_argument("Standardize: std_dev must be positive");
    }
}

Image<Float32> Standardize::operator()(Image<Float32> img) const {
    cv::Mat dst;
    img.mat().convertTo(dst, CV_32FC1, 1.0 / std_dev_, -mean_ / std_dev_);
    return Image<Float32>(std::move(dst));
}

} // namespace improc::core
