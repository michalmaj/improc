// src/core/ops/drawing.cpp
#include "improc/core/ops/drawing.hpp"

namespace improc::core {

Image<BGR> DrawText::operator()(Image<BGR> img) const {
    cv::Mat dst = img.mat().clone();
    cv::putText(dst, text_, position_, cv::FONT_HERSHEY_SIMPLEX,
                font_scale_, color_, thickness_, cv::LINE_AA);
    return Image<BGR>(std::move(dst));
}

Image<BGR> DrawLine::operator()(Image<BGR> img) const {
    cv::Mat dst = img.mat().clone();
    cv::line(dst, p1_, p2_, color_, thickness_, cv::LINE_AA);
    return Image<BGR>(std::move(dst));
}

Image<BGR> DrawCircle::operator()(Image<BGR> img) const {
    cv::Mat dst = img.mat().clone();
    cv::circle(dst, center_, radius_, color_, thickness_, cv::LINE_AA);
    return Image<BGR>(std::move(dst));
}

Image<BGR> DrawRectangle::operator()(Image<BGR> img) const {
    cv::Mat dst = img.mat().clone();
    cv::rectangle(dst, rect_, color_, thickness_, cv::LINE_AA);
    return Image<BGR>(std::move(dst));
}

} // namespace improc::core
