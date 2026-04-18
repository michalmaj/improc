// include/improc/core/ops/bilateral_filter.hpp
#pragma once

#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::core {

// Edge-preserving bilateral filter.
//
// Smooths regions of uniform colour while keeping edges sharp by weighting
// neighbours by both spatial distance (sigma_space) and colour similarity
// (sigma_color).
//
// Works on Image<Gray> and Image<BGR> (8-bit only — OpenCV requirement).
//
// Defaults: diameter=9, sigma_color=75, sigma_space=75.
// Rule of thumb: larger sigma_color → broader colour range smoothed;
//                larger sigma_space → larger spatial neighbourhood.
struct BilateralFilter {
    BilateralFilter& diameter(int d) {
        if (d <= 0)
            throw ParameterError{"diameter", "must be positive", "BilateralFilter"};
        diameter_ = d;
        return *this;
    }
    BilateralFilter& sigma_color(double s) {
        if (s <= 0.0)
            throw ParameterError{"sigma_color", "must be positive", "BilateralFilter"};
        sigma_color_ = s;
        return *this;
    }
    BilateralFilter& sigma_space(double s) {
        if (s <= 0.0)
            throw ParameterError{"sigma_space", "must be positive", "BilateralFilter"};
        sigma_space_ = s;
        return *this;
    }

    Image<Gray> operator()(Image<Gray> img) const;
    Image<BGR>  operator()(Image<BGR>  img) const;

private:
    int    diameter_    = 9;
    double sigma_color_ = 75.0;
    double sigma_space_ = 75.0;
};

} // namespace improc::core
