// src/core/ops/edge.cpp
#include "improc/core/ops/edge.hpp"

namespace improc::core {

// ── SobelEdge ────────────────────────────────────────────────────────────────

Image<Gray> SobelEdge::operator()(Image<Gray> img) const {
    cv::Mat gx, gy, magnitude;
    cv::Sobel(img.mat(), gx, CV_32F, 1, 0, ksize_);
    cv::Sobel(img.mat(), gy, CV_32F, 0, 1, ksize_);
    cv::magnitude(gx, gy, magnitude);
    cv::Mat dst;
    magnitude.convertTo(dst, CV_8U);
    return Image<Gray>(std::move(dst));
}

Image<Gray> SobelEdge::operator()(Image<BGR> img) const {
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    return (*this)(Image<Gray>(std::move(gray)));
}

// ── CannyEdge ────────────────────────────────────────────────────────────────

Image<Gray> CannyEdge::operator()(Image<Gray> img) const {
    cv::Mat dst;
    cv::Canny(img.mat(), dst, threshold1_, threshold2_, aperture_size_);
    return Image<Gray>(std::move(dst));
}

Image<Gray> CannyEdge::operator()(Image<BGR> img) const {
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    return (*this)(Image<Gray>(std::move(gray)));
}

} // namespace improc::core
