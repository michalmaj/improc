// src/core/ops/hough.cpp
#include "improc/core/ops/hough.hpp"

namespace improc::core {

HoughLinesP& HoughLinesP::rho(double r) {
    rho_ = r;
    return *this;
}

HoughLinesP& HoughLinesP::theta(double t) {
    theta_ = t;
    return *this;
}

HoughLinesP& HoughLinesP::threshold(int t) {
    threshold_ = t;
    return *this;
}

HoughLinesP& HoughLinesP::min_line_length(double l) {
    min_line_length_ = l;
    return *this;
}

HoughLinesP& HoughLinesP::max_line_gap(double g) {
    max_line_gap_ = g;
    return *this;
}

std::vector<cv::Vec4i> HoughLinesP::operator()(const Image<Gray>& img) const {
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(img.mat(), lines, rho_, theta_, threshold_,
                    min_line_length_, max_line_gap_);
    return lines;
}

HoughCircles& HoughCircles::min_dist(double d) {
    min_dist_ = d;
    return *this;
}

HoughCircles& HoughCircles::param1(double p) {
    param1_ = p;
    return *this;
}

HoughCircles& HoughCircles::param2(double p) {
    param2_ = p;
    return *this;
}

HoughCircles& HoughCircles::min_radius(int r) {
    min_radius_ = r;
    return *this;
}

HoughCircles& HoughCircles::max_radius(int r) {
    max_radius_ = r;
    return *this;
}

std::vector<cv::Vec3f> HoughCircles::operator()(const Image<Gray>& img) const {
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(img.mat(), circles, cv::HOUGH_GRADIENT, 1.0,
                     min_dist_, param1_, param2_, min_radius_, max_radius_);
    return circles;
}

} // namespace improc::core
