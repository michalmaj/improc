// include/improc/visualization/histogram.hpp
#pragma once

#include <stdexcept>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/pipeline.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;
using improc::core::Gray;
using improc::core::Float32;

struct Histogram {
    Histogram& bins(int n) {
        if (n <= 0) throw std::invalid_argument("Histogram: bins must be positive");
        bins_ = n;
        return *this;
    }
    Histogram& width(int w) {
        if (w <= 0) throw std::invalid_argument("Histogram: width must be positive");
        width_ = w;
        return *this;
    }
    Histogram& height(int h) {
        if (h <= 0) throw std::invalid_argument("Histogram: height must be positive");
        height_ = h;
        return *this;
    }

    Image<BGR> operator()(Image<BGR>     img) const;
    Image<BGR> operator()(Image<Gray>    img) const;
    Image<BGR> operator()(Image<Float32> img) const;

private:
    cv::Mat render(const std::vector<cv::Mat>& hists,
                   const std::vector<cv::Scalar>& colors) const;

    int bins_   = 256;
    int width_  = 512;
    int height_ = 256;
};

} // namespace improc::visualization
