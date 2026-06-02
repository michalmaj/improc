// src/visualization/draw_contours.cpp
#include "improc/visualization/draw_contours.hpp"
#include <opencv2/imgproc.hpp>

namespace improc::visualization {

Image<BGR> DrawContours::operator()(Image<BGR> img) const {
    cv::Mat dst = img.mat().clone();
    cv::drawContours(dst, cs_.contours, index_, color_, thickness_,
                     cv::LINE_AA, cs_.hierarchy);
    return Image<BGR>(std::move(dst));
}

} // namespace improc::visualization
