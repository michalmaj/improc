// src/visualization/histogram.cpp
#include "improc/visualization/histogram.hpp"
#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace improc::visualization {

namespace {
cv::Mat compute_hist(const cv::Mat& src, int bins,
                     float range_min, float range_max) {
    float ranges[] = {range_min, range_max};
    const float* hist_ranges[] = {ranges};
    cv::Mat hist;
    cv::calcHist(&src, 1, nullptr, cv::Mat{}, hist, 1, &bins,
                 hist_ranges, /*uniform=*/true, /*accumulate=*/false);
    return hist;
}
} // namespace

cv::Mat Histogram::render(const std::vector<cv::Mat>& hists,
                          const std::vector<cv::Scalar>& colors) const {
    cv::Mat canvas(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));
    const int bin_w = std::max(1, width_ / bins_);

    for (std::size_t c = 0; c < hists.size(); ++c) {
        cv::Mat hist = hists[c].clone();
        cv::normalize(hist, hist, 0, height_, cv::NORM_MINMAX);

        for (int i = 1; i < bins_; ++i) {
            int x0 = std::min(bin_w * (i - 1), width_ - 1);
            int x1 = std::min(bin_w * i,       width_ - 1);
            int y0 = height_ - static_cast<int>(hist.at<float>(i - 1));
            int y1 = height_ - static_cast<int>(hist.at<float>(i));
            cv::line(canvas, cv::Point(x0, y0), cv::Point(x1, y1), colors[c], 1);
        }
    }
    return canvas;
}

Image<BGR> Histogram::operator()(Image<BGR> img) const {
    std::vector<cv::Mat> channels(3);
    cv::split(img.mat(), channels);

    std::vector<cv::Mat> hists;
    for (auto& ch : channels)
        hists.push_back(compute_hist(ch, bins_, 0.0f, 256.0f));

    return Image<BGR>(render(hists, {{255,0,0}, {0,255,0}, {0,0,255}}));
}

Image<BGR> Histogram::operator()(Image<Gray> img) const {
    auto hist = compute_hist(img.mat(), bins_, 0.0f, 256.0f);
    return Image<BGR>(render({hist}, {{255, 255, 255}}));
}

Image<BGR> Histogram::operator()(Image<Float32> img) const {
    double min_val, max_val;
    cv::minMaxLoc(img.mat(), &min_val, &max_val);
    float r_min = static_cast<float>(min_val);
    float r_max = (max_val > min_val) ? static_cast<float>(max_val) : r_min + 1.0f;
    auto hist = compute_hist(img.mat(), bins_, r_min, r_max);
    return Image<BGR>(render({hist}, {{255, 255, 255}}));
}

} // namespace improc::visualization
