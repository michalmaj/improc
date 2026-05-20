// include/improc/core/ops/optical_flow.hpp
#pragma once
#include <vector>
#include <opencv2/video/tracking.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct SparseLKFlowResult {
    std::vector<cv::Point2f> points;
    std::vector<uchar>       status;
    std::vector<float>       error;
};

struct SparseLKFlow {
    SparseLKFlow& win_size(cv::Size s)  { win_size_  = s; return *this; }
    SparseLKFlow& max_level(int n)      { max_level_ = n; return *this; }
    SparseLKFlow& max_iter(int n)       { max_iter_  = n; return *this; }
    SparseLKFlow& epsilon(double e)     { epsilon_   = e; return *this; }

    SparseLKFlowResult operator()(const Image<Gray>& prev,
                                  const Image<Gray>& next,
                                  const std::vector<cv::Point2f>& prev_pts) const;

private:
    cv::Size win_size_  = {21, 21};
    int      max_level_ = 3;
    int      max_iter_  = 30;
    double   epsilon_   = 0.01;
};

struct DenseFarnebackFlow {
    DenseFarnebackFlow& pyr_scale(double s)  { pyr_scale_  = s; return *this; }
    DenseFarnebackFlow& levels(int n)        { levels_     = n; return *this; }
    DenseFarnebackFlow& win_size(int n)      { win_size_   = n; return *this; }
    DenseFarnebackFlow& iterations(int n)    { iterations_ = n; return *this; }
    DenseFarnebackFlow& poly_n(int n)        { poly_n_     = n; return *this; }
    DenseFarnebackFlow& poly_sigma(double s) { poly_sigma_ = s; return *this; }

    Image<Flow> operator()(const Image<Gray>& prev, const Image<Gray>& next) const;

private:
    double pyr_scale_  = 0.5;
    int    levels_     = 3;
    int    win_size_   = 15;
    int    iterations_ = 3;
    int    poly_n_     = 5;
    double poly_sigma_ = 1.2;
};

} // namespace improc::core
