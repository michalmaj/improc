// include/improc/core/ops/flood_fill.hpp
#pragma once
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct FloodFill {
    FloodFill& lo_diff(cv::Scalar lo) { lo_diff_ = lo; return *this; }
    FloodFill& up_diff(cv::Scalar hi) { up_diff_ = hi; return *this; }

    Image<BGR>  operator()(const Image<BGR>&  img, cv::Point seed, cv::Scalar new_color) const;
    Image<Gray> operator()(const Image<Gray>& img, cv::Point seed, uchar      new_val)   const;

private:
    cv::Scalar lo_diff_{0, 0, 0};
    cv::Scalar up_diff_{0, 0, 0};
};

} // namespace improc::core
