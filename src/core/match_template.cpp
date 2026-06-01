// src/core/match_template.cpp
#include "improc/core/ops/match_template.hpp"
#include "improc/exceptions.hpp"
#include <stdexcept>

namespace improc::core {

MatchTemplate& MatchTemplate::method(int m) {
    method_ = m;
    return *this;
}

std::pair<cv::Point, double> MatchTemplate::operator()(const Image<BGR>& img,
                                                        const Image<BGR>& templ) const {
    if (templ.mat().cols > img.mat().cols || templ.mat().rows > img.mat().rows)
        throw improc::ParameterError{"templ", "must not be larger than the image", "MatchTemplate"};

    cv::Mat result;
    cv::matchTemplate(img.mat(), templ.mat(), result, method_);

    double min_val, max_val;
    cv::Point min_loc, max_loc;
    cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

    if (method_ == cv::TM_SQDIFF || method_ == cv::TM_SQDIFF_NORMED)
        return {min_loc, min_val};
    return {max_loc, max_val};
}

} // namespace improc::core
