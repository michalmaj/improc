// include/improc/visualization/iou_histogram.hpp
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <numeric>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;

namespace detail::iou {
inline const cv::Scalar kBg    {42,  23,  15};
inline const cv::Scalar kPanel {59,  41,  30};
inline const cv::Scalar kText  {184, 163, 148};
inline const cv::Scalar kMuted {139, 116, 100};
inline const cv::Scalar kAccent{250, 139, 167};
inline const cv::Scalar kAmber { 11, 158, 245};
inline const cv::Scalar kGreen {153, 211,  52};
// Violet gradient for above-threshold bars
inline const std::array<cv::Scalar,4> kViolets{
    cv::Scalar{237,  62, 124},
    cv::Scalar{217,  40, 109},
    cv::Scalar{197,  29,  92},
    cv::Scalar{149,  29,  76},
};
} // namespace detail::iou

struct IoUHistogram {
    explicit IoUHistogram(std::vector<float> scores)
        : scores_(std::move(scores)) {}

    IoUHistogram& width(int w)         { width_ = w;            return *this; }
    IoUHistogram& height(int h)        { height_ = h;           return *this; }
    IoUHistogram& bins(int b)          { bins_ = b;             return *this; }
    IoUHistogram& threshold(float t)   { threshold_ = t;        return *this; }
    IoUHistogram& title(std::string t) { title_ = std::move(t); return *this; }

    Image<BGR> operator()() const {
        if (width_ <= 0)
            throw improc::ParameterError{"width",     "must be positive",   "IoUHistogram"};
        if (height_ <= 0)
            throw improc::ParameterError{"height",    "must be positive",   "IoUHistogram"};
        if (bins_ <= 0)
            throw improc::ParameterError{"bins",      "must be positive",   "IoUHistogram"};
        if (threshold_ < 0.f || threshold_ > 1.f)
            throw improc::ParameterError{"threshold", "must be in [0, 1]",  "IoUHistogram"};

        cv::Mat canvas(height_, width_, CV_8UC3, detail::iou::kBg);
        if (scores_.empty()) return Image<BGR>(std::move(canvas));

        const int ml = 52, mr = 14, mt = 28, mb = 46;
        const int pw = width_ - ml - mr;
        const int ph = height_ - mt - mb;

        // Build histogram counts
        std::vector<int> counts(bins_, 0);
        for (float s : scores_) {
            int bi = std::min(bins_-1, static_cast<int>(s * bins_));
            if (bi >= 0) ++counts[bi];
        }
        int max_count = *std::max_element(counts.begin(), counts.end());
        if (max_count == 0) max_count = 1;

        // Axes
        cv::line(canvas, {ml, mt},    {ml, mt+ph},    detail::iou::kPanel, 1);
        cv::line(canvas, {ml, mt+ph}, {ml+pw, mt+ph}, detail::iou::kPanel, 1);

        // Threshold vertical dashed line
        int thr_x = ml + static_cast<int>(threshold_ * pw);
        for (int y = mt; y < mt+ph; y += 8)
            cv::line(canvas, {thr_x, y}, {thr_x, std::min(y+4, mt+ph)},
                     detail::iou::kAmber, 1, cv::LINE_AA);
        auto thr_txt = std::format("thr={:.2f}", threshold_);
        cv::rectangle(canvas, {thr_x+2, mt+2}, {thr_x+50, mt+14},
                      detail::iou::kPanel, cv::FILLED);
        cv::putText(canvas, thr_txt, {thr_x+4, mt+12},
                    cv::FONT_HERSHEY_SIMPLEX, 0.26, detail::iou::kAmber, 1, cv::LINE_AA);

        // Bars
        float bar_w_f = static_cast<float>(pw) / bins_;
        int   bar_w   = std::max(1, static_cast<int>(bar_w_f) - 1);

        for (int b = 0; b < bins_; ++b) {
            float bin_center = (b + 0.5f) / bins_;
            bool above = bin_center >= threshold_;
            int bh = static_cast<int>(static_cast<float>(counts[b]) / max_count * ph);
            if (bh == 0 && counts[b] > 0) bh = 1;
            int bx = ml + static_cast<int>(b * bar_w_f);
            int by = mt + ph - bh;

            cv::Scalar col;
            if (above) {
                float t = (bin_center - threshold_) / (1.0f - threshold_ + 1e-6f);
                int vi = std::min(3, static_cast<int>(t * 4));
                col = detail::iou::kViolets[vi];
            } else {
                float t = bin_center / (threshold_ + 1e-6f);
                float s = 0.3f + 0.7f * t;
                col = cv::Scalar(
                    static_cast<int>(detail::iou::kAmber[0] * s + detail::iou::kBg[0] * (1-s)),
                    static_cast<int>(detail::iou::kAmber[1] * s + detail::iou::kBg[1] * (1-s)),
                    static_cast<int>(detail::iou::kAmber[2] * s + detail::iou::kBg[2] * (1-s))
                );
            }
            cv::rectangle(canvas, {bx, by}, {bx+bar_w, mt+ph}, col, cv::FILLED);
        }

        // X axis labels
        for (int i = 0; i <= 4; ++i) {
            float v = i * 0.25f;
            int x = ml + static_cast<int>(v * pw);
            cv::Scalar tc = (std::abs(v - threshold_) < 0.13f)
                ? detail::iou::kAmber : detail::iou::kText;
            cv::putText(canvas, std::format("{:.2f}", v), {x-8, mt+ph+12},
                        cv::FONT_HERSHEY_SIMPLEX, 0.26, tc, 1, cv::LINE_AA);
        }
        cv::putText(canvas, "IoU score",
                    {ml + pw/2 - 20, mt+ph+26},
                    cv::FONT_HERSHEY_SIMPLEX, 0.30, detail::iou::kMuted, 1, cv::LINE_AA);

        // Stat badges (mean IoU + above-threshold %)
        float mean_iou = std::accumulate(scores_.begin(), scores_.end(), 0.f) / scores_.size();
        int above_count = static_cast<int>(
            std::count_if(scores_.begin(), scores_.end(),
                          [&](float s){ return s >= threshold_; }));
        float pct = 100.f * above_count / static_cast<float>(scores_.size());

        int bw2 = pw / 2 - 4;
        cv::rectangle(canvas, {ml, mt+ph+30}, {ml+bw2, height_-2},
                      detail::iou::kPanel, cv::FILLED);
        cv::putText(canvas,
                    std::format("mean IoU: {:.3f}", mean_iou),
                    {ml+6, mt+ph+42},
                    cv::FONT_HERSHEY_SIMPLEX, 0.28, detail::iou::kAccent, 1, cv::LINE_AA);
        cv::rectangle(canvas, {ml+bw2+4, mt+ph+30}, {ml+pw, height_-2},
                      detail::iou::kPanel, cv::FILLED);
        cv::putText(canvas,
                    std::format("above thr: {:.1f}%", pct),
                    {ml+bw2+10, mt+ph+42},
                    cv::FONT_HERSHEY_SIMPLEX, 0.28, detail::iou::kGreen, 1, cv::LINE_AA);

        if (!title_.empty())
            cv::putText(canvas, title_, {ml, mt-8},
                        cv::FONT_HERSHEY_SIMPLEX, 0.38, detail::iou::kText, 1, cv::LINE_AA);

        return Image<BGR>(std::move(canvas));
    }

private:
    std::vector<float> scores_;
    int         width_     = 800;
    int         height_    = 300;
    int         bins_      = 20;
    float       threshold_ = 0.5f;
    std::string title_;
};

} // namespace improc::visualization
