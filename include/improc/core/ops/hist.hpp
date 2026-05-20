// include/improc/core/ops/hist.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct CalcHist {
    CalcHist& bins(int b);
    CalcHist& range(float lo, float hi);

    cv::Mat operator()(const Image<Gray>& img) const;
    cv::Mat operator()(const Image<BGR>& img) const;

private:
    int bins_ = 256;
    float range_lo_ = 0.0f;
    float range_hi_ = 256.0f;
};

struct CompareHist {
    CompareHist& method(int m);

    double operator()(const cv::Mat& h1, const cv::Mat& h2) const;

private:
    int method_ = cv::HISTCMP_CORREL;
};

} // namespace improc::core
