// src/core/ops/hist.cpp
#include "improc/core/ops/hist.hpp"

namespace improc::core {

CalcHist& CalcHist::bins(int b) {
    bins_ = b;
    return *this;
}

CalcHist& CalcHist::range(float lo, float hi) {
    range_lo_ = lo;
    range_hi_ = hi;
    return *this;
}

cv::Mat CalcHist::operator()(const Image<Gray>& img) const {
    cv::Mat hist;
    const float range[] = {range_lo_, range_hi_};
    const float* ranges[] = {range};
    int ch = 0;
    cv::calcHist(&img.mat(), 1, &ch, cv::noArray(),
                 hist, 1, &bins_, ranges);
    return hist;
}

cv::Mat CalcHist::operator()(const Image<BGR>& img) const {
    const float range[] = {range_lo_, range_hi_};
    const float* ranges[] = {range};

    cv::Mat hists[3];
    for (int ch = 0; ch < 3; ++ch) {
        cv::calcHist(&img.mat(), 1, &ch, cv::noArray(),
                     hists[ch], 1, &bins_, ranges);
    }
    cv::Mat stacked;
    cv::vconcat(hists, 3, stacked);
    return stacked;
}

CompareHist& CompareHist::method(int m) {
    method_ = m;
    return *this;
}

double CompareHist::operator()(const cv::Mat& h1, const cv::Mat& h2) const {
    return cv::compareHist(h1, h2, method_);
}

} // namespace improc::core
