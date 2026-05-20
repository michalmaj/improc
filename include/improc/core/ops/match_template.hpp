// include/improc/core/ops/match_template.hpp
#pragma once

#include <utility>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"

namespace improc::core {

struct MatchTemplate {
    MatchTemplate& method(int m);

    std::pair<cv::Point, double> operator()(const Image<BGR>& img,
                                             const Image<BGR>& templ) const;

private:
    int method_ = cv::TM_CCOEFF_NORMED;
};

} // namespace improc::core
