// src/core/ops/haar_detect.cpp
#include "improc/core/ops/haar_detect.hpp"
#include <opencv2/imgproc.hpp>

namespace improc::core {

std::vector<cv::Rect> DetectHaar::operator()(
    const Image<BGR>& img, const cv::CascadeClassifier& cc) const
{
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Rect> rects;
    const_cast<cv::CascadeClassifier&>(cc).detectMultiScale(
        gray, rects, scale_factor_, min_neighbors_, 0, min_size_, max_size_);
    return rects;
}

} // namespace improc::core
