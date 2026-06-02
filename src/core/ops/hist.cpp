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

HistogramData CalcHist::operator()(const Image<Gray>& img) const {
    cv::Mat hist;
    const float range[] = {range_lo_, range_hi_};
    const float* ranges[] = {range};
    int ch = 0;
    cv::calcHist(&img.mat(), 1, &ch, cv::noArray(),
                 hist, 1, &bins_, ranges);
    return HistogramData{hist, bins_, range_lo_, range_hi_, 1};
}

HistogramData CalcHist::operator()(const Image<BGR>& img) const {
    const float range[] = {range_lo_, range_hi_};
    const float* ranges[] = {range};
    cv::Mat combined(bins_, 3, CV_32F);
    for (int c = 0; c < 3; ++c) {
        cv::Mat col_hist;
        cv::calcHist(&img.mat(), 1, &c, cv::noArray(),
                     col_hist, 1, &bins_, ranges);
        col_hist.copyTo(combined.col(c));
    }
    return HistogramData{combined, bins_, range_lo_, range_hi_, 3};
}

CompareHist& CompareHist::method(int m) {
    method_ = m;
    return *this;
}

double CompareHist::operator()(const HistogramData& h1, const HistogramData& h2) const {
    return cv::compareHist(h1.data, h2.data, method_);
}

} // namespace improc::core
