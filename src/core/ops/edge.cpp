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

// ── LaplacianEdge ─────────────────────────────────────────────────────────────

Image<Gray> LaplacianEdge::operator()(Image<Gray> img) const {
    cv::Mat dst16;
    cv::Laplacian(img.mat(), dst16, CV_16S, ksize_, scale_, delta_);
    cv::Mat dst8;
    cv::convertScaleAbs(dst16, dst8);
    return Image<Gray>(std::move(dst8));
}

Image<Gray> LaplacianEdge::operator()(Image<BGR> img) const {
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    return (*this)(Image<Gray>(std::move(gray)));
}

// ── HarrisCorner ──────────────────────────────────────────────────────────────

Image<Gray> HarrisCorner::operator()(Image<Gray> img) const {
    cv::Mat response;
    cv::cornerHarris(img.mat(), response, block_size_, ksize_, k_);
    cv::normalize(response, response, 0, 255, cv::NORM_MINMAX);
    cv::Mat dst;
    response.convertTo(dst, CV_8U);
    return Image<Gray>(std::move(dst));
}

Image<Gray> HarrisCorner::operator()(Image<BGR> img) const {
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    return (*this)(Image<Gray>(std::move(gray)));
}

} // namespace improc::core
