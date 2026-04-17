// include/improc/visualization/histogram.hpp
#pragma once

#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;
using improc::core::Gray;
using improc::core::Float32;

struct Histogram {
    Histogram& bins(int n) {
        if (n <= 0) throw ParameterError{"bins", "must be positive", "Histogram"};
        bins_ = n;
        return *this;
    }
    Histogram& width(int w) {
        if (w <= 0) throw ParameterError{"width", "must be positive", "Histogram"};
        width_ = w;
        return *this;
    }
    Histogram& height(int h) {
        if (h <= 0) throw ParameterError{"height", "must be positive", "Histogram"};
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
