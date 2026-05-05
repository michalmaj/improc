// src/core/ops/contours.cpp
#include "improc/core/ops/contours.hpp"

namespace improc::core {

namespace {

int to_cv_mode(FindContours::Mode m) {
    switch (m) {
        case FindContours::Mode::External: return cv::RETR_EXTERNAL;
        case FindContours::Mode::List:     return cv::RETR_LIST;
        case FindContours::Mode::CComp:    return cv::RETR_CCOMP;
        case FindContours::Mode::Tree:     return cv::RETR_TREE;
    }
    std::unreachable();
}

int to_cv_method(FindContours::Method m) {
    switch (m) {
        case FindContours::Method::None:    return cv::CHAIN_APPROX_NONE;
        case FindContours::Method::Simple:  return cv::CHAIN_APPROX_SIMPLE;
        case FindContours::Method::TehChin: return cv::CHAIN_APPROX_TC89_L1;
    }
    std::unreachable();
}

} // namespace

ContourSet FindContours::operator()(Image<Gray> img) const {
    cv::Mat src = img.mat().clone();
    ContourSet result;
    cv::findContours(src, result.contours, result.hierarchy,
                     to_cv_mode(mode_), to_cv_method(method_));
    return result;
}

Image<BGR> DrawContours::operator()(Image<BGR> img) const {
    cv::Mat dst = img.mat().clone();
    cv::drawContours(dst, cs_.contours, index_, color_, thickness_,
                     cv::LINE_AA, cs_.hierarchy);
    return Image<BGR>(std::move(dst));
}

} // namespace improc::core
