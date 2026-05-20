#pragma once
#include <stdexcept>
#include <vector>
#include <opencv2/video/tracking.hpp>
#include "improc/core/image.hpp"
#include "improc/core/format_traits.hpp"

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

} // namespace improc::core
