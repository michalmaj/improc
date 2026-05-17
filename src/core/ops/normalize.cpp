// src/core/ops/normalize.cpp
#include "improc/core/ops/normalize.hpp"
#include <cmath>
#include <opencv2/core.hpp>
#include "improc/exceptions.hpp"

using improc::ParameterError;

namespace improc::core {

namespace {
// Returns global min/max across all channels by reshaping to single-channel.
void global_min_max(const cv::Mat& mat, double& min_val, double& max_val) {
    cv::minMaxLoc(mat.reshape(1), &min_val, &max_val);
}
} // namespace

Image<Float32> Normalize::operator()(Image<Float32> img) const {
    double min_val, max_val;
    global_min_max(img.mat(), min_val, max_val);
    if (std::abs(max_val - min_val) < 1e-10) {
        img.mat().setTo(0.0f);
    } else {
        const double scale = 1.0 / (max_val - min_val);
        const double shift = -min_val * scale;
        img.mat().convertTo(img.mat(), CV_32FC1, scale, shift);
    }
    return img;
}

Image<Float32C3> Normalize::operator()(Image<Float32C3> img) const {
    double min_val, max_val;
    global_min_max(img.mat(), min_val, max_val);
    if (std::abs(max_val - min_val) < 1e-10) {
        img.mat().setTo(0.0f);
    } else {
        const double scale = 1.0 / (max_val - min_val);
        const double shift = -min_val * scale;
        img.mat().convertTo(img.mat(), CV_32FC3, scale, shift);
    }
    return img;
}

NormalizeTo::NormalizeTo(float min, float max) : min_(min), max_(max) {
    if (min >= max)
        throw ParameterError{"min", "must be less than max", "NormalizeTo"};
}

Image<Float32> NormalizeTo::operator()(Image<Float32> img) const {
    double min_val, max_val;
    global_min_max(img.mat(), min_val, max_val);
    if (std::abs(max_val - min_val) < 1e-10) {
        img.mat().setTo(0.0f);
    } else {
        const double scale = (max_ - min_) / (max_val - min_val);
        const double shift = min_ - min_val * scale;
        img.mat().convertTo(img.mat(), CV_32FC1, scale, shift);
    }
    return img;
}

Image<Float32C3> NormalizeTo::operator()(Image<Float32C3> img) const {
    double min_val, max_val;
    global_min_max(img.mat(), min_val, max_val);
    if (std::abs(max_val - min_val) < 1e-10) {
        img.mat().setTo(0.0f);
    } else {
        const double scale = (max_ - min_) / (max_val - min_val);
        const double shift = min_ - min_val * scale;
        img.mat().convertTo(img.mat(), CV_32FC3, scale, shift);
    }
    return img;
}

Standardize::Standardize(float mean, float std_dev) : mean_(mean), std_dev_(std_dev) {
    if (std_dev <= 0.0f)
        throw ParameterError{"std_dev", "must be positive", "Standardize"};
}

Image<Float32> Standardize::operator()(Image<Float32> img) const {
    img.mat().convertTo(img.mat(), CV_32FC1, 1.0 / std_dev_, -mean_ / std_dev_);
    return img;
}

Image<Float32C3> Standardize::operator()(Image<Float32C3> img) const {
    img.mat().convertTo(img.mat(), CV_32FC3, 1.0 / std_dev_, -mean_ / std_dev_);
    return img;
}

} // namespace improc::core
