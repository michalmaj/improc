// src/core/convert.cpp
#include "improc/core/convert.hpp"
#include <opencv2/imgproc.hpp>
#include <utility>

namespace improc::core {

template<>
Image<Gray> convert<Gray, BGR>(const Image<BGR>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_BGR2GRAY);
    return Image<Gray>(std::move(dst));
}

template<>
Image<BGR> convert<BGR, Gray>(const Image<Gray>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_GRAY2BGR);
    return Image<BGR>(std::move(dst));
}

template<>
Image<BGRA> convert<BGRA, BGR>(const Image<BGR>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_BGR2BGRA);
    return Image<BGRA>(std::move(dst));
}

template<>
Image<BGR> convert<BGR, BGRA>(const Image<BGRA>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_BGRA2BGR);
    return Image<BGR>(std::move(dst));
}

template<>
Image<Float32> convert<Float32, Gray>(const Image<Gray>& src) {
    cv::Mat dst;
    src.mat().convertTo(dst, CV_32FC1, 1.0 / 255.0);
    return Image<Float32>(std::move(dst));
}

template<>
Image<Float32C3> convert<Float32C3, BGR>(const Image<BGR>& src) {
    cv::Mat dst;
    src.mat().convertTo(dst, CV_32FC3, 1.0 / 255.0);
    return Image<Float32C3>(std::move(dst));
}

template<>
Image<Gray> convert<Gray, Float32>(const Image<Float32>& src) {
    cv::Mat dst;
    src.mat().convertTo(dst, CV_8UC1, 255.0);
    return Image<Gray>(std::move(dst));
}

template<>
Image<BGR> convert<BGR, Float32C3>(const Image<Float32C3>& src) {
    cv::Mat dst;
    src.mat().convertTo(dst, CV_8UC3, 255.0);
    return Image<BGR>(std::move(dst));
}

template<>
Image<HSV> convert<HSV, BGR>(const Image<BGR>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_BGR2HSV);
    return Image<HSV>(std::move(dst));
}

template<>
Image<BGR> convert<BGR, HSV>(const Image<HSV>& src) {
    cv::Mat dst;
    cv::cvtColor(src.mat(), dst, cv::COLOR_HSV2BGR);
    return Image<BGR>(std::move(dst));
}

} // namespace improc::core
