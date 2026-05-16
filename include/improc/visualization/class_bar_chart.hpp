// include/improc/visualization/class_bar_chart.hpp
#pragma once

#include <array>
#include <format>
#include <map>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"
#include "improc/ml/eval/classification.hpp"
#include "improc/ml/eval/detection.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;

namespace detail::cbc {
inline const cv::Scalar kBg    {42,  23,  15};
inline const cv::Scalar kPanel {59,  41,  30};
inline const cv::Scalar kText  {184, 163, 148};
// Bar colors: Precision=violet, Recall=cyan, F1=green
inline const cv::Scalar kViolet{250, 139, 167};
inline const cv::Scalar kCyan  {248, 189,  56};
inline const cv::Scalar kGreen {153, 211,  52};
} // namespace detail::cbc

struct ClassBarChart {
    // From ClassMetrics: 3 bars per class (P/R/F1)
    explicit ClassBarChart(const improc::ml::ClassMetrics& m) {
        for (const auto& [cls, p] : m.per_class_precision) {
            float r = m.per_class_recall.count(cls) ? m.per_class_recall.at(cls) : 0.f;
            float f = m.per_class_f1.count(cls)     ? m.per_class_f1.at(cls)     : 0.f;
            data_[cls] = {p, r, f};
        }
    }

    // From DetectionMetrics: 1 bar per class (AP only)
    explicit ClassBarChart(const improc::ml::DetectionMetrics& m) : single_mode_(true) {
        for (const auto& [cls, ap] : m.per_class_AP)
            data_[cls] = {ap, 0.f, 0.f};
    }

    // Raw form: explicit {P, R, F1} per class
    explicit ClassBarChart(std::map<std::string, std::array<float,3>> vals) {
        for (const auto& [cls, arr] : vals)
            data_[cls] = {arr[0], arr[1], arr[2]};
    }

    ClassBarChart& width(int w)         { width_ = w;            return *this; }
    ClassBarChart& height(int h)        { height_ = h;           return *this; }
    ClassBarChart& title(std::string t) { title_ = std::move(t); return *this; }

    Image<BGR> operator()() const {
        if (width_ <= 0)  throw improc::ParameterError{"width",  "must be positive", "ClassBarChart"};
        if (height_ <= 0) throw improc::ParameterError{"height", "must be positive", "ClassBarChart"};

        cv::Mat canvas(height_, width_, CV_8UC3, detail::cbc::kBg);
        if (data_.empty()) return Image<BGR>(std::move(canvas));

        const int ml = 48, mr = 12, mt = 28, mb = 48;
        const int pw = width_ - ml - mr;
        const int ph = height_ - mt - mb;

        // Axes
        cv::line(canvas, {ml, mt},    {ml, mt+ph},    detail::cbc::kPanel, 1);
        cv::line(canvas, {ml, mt+ph}, {ml+pw, mt+ph}, detail::cbc::kPanel, 1);
        // Dashed grid at 0.5
        for (int x = ml; x < ml+pw; x += 8)
            cv::line(canvas, {x, mt+ph/2}, {std::min(x+4, ml+pw), mt+ph/2},
                     detail::cbc::kPanel, 1);

        // Y labels
        for (int i = 0; i <= 4; ++i) {
            float v = i * 0.25f;
            int y = mt + ph - static_cast<int>(v * ph);
            cv::putText(canvas, std::format("{:.2f}", v), {ml-40, y+4},
                        cv::FONT_HERSHEY_SIMPLEX, 0.26, detail::cbc::kText, 1, cv::LINE_AA);
        }

        int n_cls = static_cast<int>(data_.size());
        int bars_per_group = single_mode_ ? 1 : 3;
        int group_w = pw / n_cls;
        int bar_w   = std::max(4, (group_w - 4) / (bars_per_group + 1));
        int group_pad = (group_w - bars_per_group * bar_w) / 2;

        const std::array<cv::Scalar, 3> bar_colors{
            detail::cbc::kViolet, detail::cbc::kCyan, detail::cbc::kGreen
        };

        int gi = 0;
        for (const auto& [cls, vals] : data_) {
            int gx = ml + gi * group_w;
            int bars = single_mode_ ? 1 : 3;
            for (int b = 0; b < bars; ++b) {
                float v = std::min(1.0f, std::max(0.0f, vals[b]));
                int bx  = gx + group_pad + b * (bar_w + 2);
                int bh  = static_cast<int>(v * ph);
                int by  = mt + ph - bh;
                cv::rectangle(canvas, {bx, by}, {bx+bar_w, mt+ph},
                              bar_colors[b], cv::FILLED);
            }
            // Class label below group
            int lx = gx + group_w/2 - 10;
            cv::putText(canvas, cls, {lx, mt+ph+14},
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, detail::cbc::kText, 1, cv::LINE_AA);
            ++gi;
        }

        // Legend box (top-right)
        if (!single_mode_) {
            const std::array<std::string,3> labels{"Precision","Recall","F1"};
            int lw = 74, lh = 46, lx = ml+pw-lw-4, ly = mt+6;
            cv::rectangle(canvas, {lx, ly}, {lx+lw, ly+lh}, detail::cbc::kPanel, cv::FILLED);
            for (int i = 0; i < 3; ++i) {
                int ey = ly + 8 + i*14;
                cv::rectangle(canvas, {lx+5, ey}, {lx+13, ey+8},
                              bar_colors[i], cv::FILLED);
                cv::putText(canvas, labels[i], {lx+16, ey+8},
                            cv::FONT_HERSHEY_SIMPLEX, 0.26, detail::cbc::kText, 1, cv::LINE_AA);
            }
        } else {
            // Single-mode: "AP" label in legend
            cv::rectangle(canvas, {ml+pw-40, mt+6}, {ml+pw-4, mt+22},
                          detail::cbc::kPanel, cv::FILLED);
            cv::rectangle(canvas, {ml+pw-38, mt+9}, {ml+pw-30, mt+17},
                          detail::cbc::kViolet, cv::FILLED);
            cv::putText(canvas, "AP", {ml+pw-27, mt+18},
                        cv::FONT_HERSHEY_SIMPLEX, 0.28, detail::cbc::kText, 1, cv::LINE_AA);
        }

        if (!title_.empty())
            cv::putText(canvas, title_, {ml, mt-8},
                        cv::FONT_HERSHEY_SIMPLEX, 0.38, detail::cbc::kText, 1, cv::LINE_AA);

        return Image<BGR>(std::move(canvas));
    }

private:
    std::map<std::string, std::array<float,3>> data_;
    bool        single_mode_ = false;
    int         width_  = 640;
    int         height_ = 400;
    std::string title_;
};

} // namespace improc::visualization
