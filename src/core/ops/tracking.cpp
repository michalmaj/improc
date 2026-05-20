// src/core/ops/tracking.cpp
#include "improc/core/ops/tracking.hpp"

namespace improc::core {

namespace {
void check_window(const cv::Rect& window, int rows, int cols) {
    if (window.x < 0 || window.y < 0 ||
        window.x + window.width  > cols ||
        window.y + window.height > rows)
        throw std::invalid_argument("CamShift/MeanShift: window is outside image bounds");
}
} // namespace

CamShiftResult CamShift::operator()(const Image<Gray>& back_proj, cv::Rect& window) const {
    check_window(window, back_proj.rows(), back_proj.cols());
    auto criteria = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                      max_iter_, epsilon_);
    CamShiftResult result;
    result.object     = cv::CamShift(back_proj.mat(), window, criteria);
    result.iterations = max_iter_;  // CamShift doesn't expose iteration count directly
    return result;
}

int MeanShift::operator()(const Image<Gray>& back_proj, cv::Rect& window) const {
    check_window(window, back_proj.rows(), back_proj.cols());
    auto criteria = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                                      max_iter_, epsilon_);
    return cv::meanShift(back_proj.mat(), window, criteria);
}

} // namespace improc::core
