// include/improc/core/ops/tracking.hpp
#pragma once
#include <stdexcept>
#include <opencv2/video/tracking.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct CamShiftResult {
    cv::RotatedRect object;
    int             iterations;
};

struct CamShift {
    CamShift& epsilon(double e)  { epsilon_  = e; return *this; }
    CamShift& max_iter(int n)    { max_iter_ = n; return *this; }

    CamShiftResult operator()(const Image<Gray>& back_proj, cv::Rect& window) const;

private:
    double epsilon_  = 1.0;
    int    max_iter_ = 10;
};

struct MeanShift {
    MeanShift& epsilon(double e)  { epsilon_  = e; return *this; }
    MeanShift& max_iter(int n)    { max_iter_ = n; return *this; }

    int operator()(const Image<Gray>& back_proj, cv::Rect& window) const;

private:
    double epsilon_  = 1.0;
    int    max_iter_ = 10;
};

} // namespace improc::core
