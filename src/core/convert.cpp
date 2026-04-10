// src/core/convert.cpp
#include "improc/core/convert.hpp"
#include <opencv2/imgproc.hpp>

namespace improc::core {

template<>
Image<Gray> convert<Gray, BGR>(const Image<BGR>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_BGR2GRAY);
    return Image<Gray>(dst);
}

template<>
Image<BGR> convert<BGR, Gray>(const Image<Gray>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_GRAY2BGR);
    return Image<BGR>(dst);
}

template<>
Image<BGRA> convert<BGRA, BGR>(const Image<BGR>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_BGR2BGRA);
    return Image<BGRA>(dst);
}

template<>
Image<BGR> convert<BGR, BGRA>(const Image<BGRA>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_BGRA2BGR);
    return Image<BGR>(dst);
}

template<>
Image<Float32> convert<Float32, Gray>(const Image<Gray>& src) {
    cv::Mat dst;
    src.mat().convertTo(dst, CV_32FC1, 1.0 / 255.0);
    return Image<Float32>(dst);
}

} // namespace improc::core
