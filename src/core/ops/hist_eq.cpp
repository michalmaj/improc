// src/core/ops/hist_eq.cpp
#include "improc/core/ops/hist_eq.hpp"

namespace improc::core {

Image<Gray> HistogramEqualization::operator()(Image<Gray> img) const {
    cv::Mat dst;
    cv::equalizeHist(img.mat(), dst);
    return Image<Gray>(std::move(dst));
}

Image<BGR> HistogramEqualization::operator()(Image<BGR> img) const {
    cv::Mat ycrcb;
    cv::cvtColor(img.mat(), ycrcb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);
    cv::equalizeHist(channels[0], channels[0]);  // equalize Y channel only
    cv::merge(channels, ycrcb);

    cv::Mat dst;
    cv::cvtColor(ycrcb, dst, cv::COLOR_YCrCb2BGR);
    return Image<BGR>(std::move(dst));
}

} // namespace improc::core
