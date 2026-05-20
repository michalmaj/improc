// include/improc/core/ops/hough.hpp
#pragma once

#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct HoughLinesP {
    HoughLinesP& rho(double r);
    HoughLinesP& theta(double t);
    HoughLinesP& threshold(int t);
    HoughLinesP& min_line_length(double l);
    HoughLinesP& max_line_gap(double g);

    std::vector<cv::Vec4i> operator()(const Image<Gray>& img) const;

private:
    double rho_             = 1.0;
    double theta_           = CV_PI / 180.0;
    int    threshold_       = 80;
    double min_line_length_ = 0.0;
    double max_line_gap_    = 0.0;
};

struct HoughCircles {
    HoughCircles& min_dist(double d);
    HoughCircles& param1(double p);
    HoughCircles& param2(double p);
    HoughCircles& min_radius(int r);
    HoughCircles& max_radius(int r);

    std::vector<cv::Vec3f> operator()(const Image<Gray>& img) const;

private:
    double min_dist_   = 20.0;
    double param1_     = 100.0;
    double param2_     = 30.0;
    int    min_radius_ = 0;
    int    max_radius_ = 0;
};

} // namespace improc::core
