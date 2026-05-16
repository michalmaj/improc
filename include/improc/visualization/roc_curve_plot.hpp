// include/improc/visualization/roc_curve_plot.hpp
#pragma once

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;

namespace detail::roc {
inline const cv::Scalar kBg    {42,  23,  15};
inline const cv::Scalar kPanel {59,  41,  30};
inline const cv::Scalar kText  {184, 163, 148};
inline const cv::Scalar kMuted {139, 116, 100};
inline const std::vector<cv::Scalar> kColors{
    {250, 139, 167}, {153, 211,  52}, { 11, 158, 245}, {248, 189,  56}
};
} // namespace detail::roc

struct ROCCurvePlot {
    using RocMap = std::map<std::string,
        std::pair<std::vector<float>, std::vector<float>>>;

    // Map form: class → (fpr[], tpr[])
    explicit ROCCurvePlot(RocMap curves) : curves_(std::move(curves)) {}

    // Raw form: separate fpr and tpr maps
    ROCCurvePlot(std::map<std::string, std::vector<float>> fprs,
                 std::map<std::string, std::vector<float>> tprs) {
        for (const auto& [cls, fpr] : fprs) {
            auto it = tprs.find(cls);
            if (it != tprs.end()) curves_[cls] = {fpr, it->second};
        }
    }

    ROCCurvePlot& width(int w)          { width_ = w;            return *this; }
    ROCCurvePlot& height(int h)         { height_ = h;           return *this; }
    ROCCurvePlot& title(std::string t)  { title_ = std::move(t); return *this; }

    Image<BGR> operator()() const {
        if (width_ <= 0)  throw improc::ParameterError{"width",  "must be positive", "ROCCurvePlot"};
        if (height_ <= 0) throw improc::ParameterError{"height", "must be positive", "ROCCurvePlot"};

        cv::Mat canvas(height_, width_, CV_8UC3, detail::roc::kBg);

        const int ml = 48, mr = 12, mt = 28, mb = 48;
        const int pw = width_ - ml - mr;
        const int ph = height_ - mt - mb;

        // Axes
        cv::line(canvas, {ml, mt},    {ml, mt+ph},    detail::roc::kPanel, 1);
        cv::line(canvas, {ml, mt+ph}, {ml+pw, mt+ph}, detail::roc::kPanel, 1);

        // Dashed diagonal (random classifier)
        for (int i = 0; i < 20; ++i) {
            float t0 = static_cast<float>(i)   / 20.f;
            float t1 = static_cast<float>(i+1) / 20.f;
            if (i % 2 == 0) {
                int x0 = ml + static_cast<int>(t0 * pw);
                int y0 = mt + ph - static_cast<int>(t0 * ph);
                int x1 = ml + static_cast<int>(t1 * pw);
                int y1 = mt + ph - static_cast<int>(t1 * ph);
                cv::line(canvas, {x0,y0}, {x1,y1}, detail::roc::kPanel, 1, cv::LINE_AA);
            }
        }

        // Y/X axis labels
        auto yl = [&](float v, int y) {
            cv::putText(canvas, std::format("{:.1f}", v), {ml-38, y+4},
                        cv::FONT_HERSHEY_SIMPLEX, 0.28, detail::roc::kText, 1, cv::LINE_AA);
        };
        yl(1.0f, mt);  yl(0.5f, mt + ph/2);  yl(0.0f, mt + ph);
        cv::putText(canvas, "FPR", {ml + pw/2 - 10, mt+ph+32},
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, detail::roc::kMuted, 1, cv::LINE_AA);
        cv::putText(canvas, "TPR", {4, mt + ph/2 + 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, detail::roc::kMuted, 1, cv::LINE_AA);

        // Coordinate mapping
        auto sx = [&](float fpr) { return ml + static_cast<int>(fpr * pw); };
        auto sy = [&](float tpr) { return mt + ph - static_cast<int>(tpr * ph); };

        // Curves + AUC per class (trapezoid rule)
        std::map<std::string, float> aucs;
        int ci = 0;
        for (const auto& [cls, vecs] : curves_) {
            const auto& [fpr, tpr] = vecs;
            if (fpr.empty()) { ++ci; continue; }
            cv::Scalar col = detail::roc::kColors[ci % static_cast<int>(detail::roc::kColors.size())];
            std::vector<cv::Point> pts;
            pts.reserve(fpr.size());
            for (std::size_t i = 0; i < fpr.size(); ++i)
                pts.push_back({sx(fpr[i]), sy(tpr[i])});
            cv::polylines(canvas, pts, false, col, 2, cv::LINE_AA);
            // AUC via trapezoid
            float auc = 0.f;
            for (std::size_t i = 1; i < fpr.size(); ++i)
                auc += std::abs(fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) * 0.5f;
            aucs[cls] = auc;
            ++ci;
        }

        // Legend box (bottom-right, has AUC values)
        if (!curves_.empty()) {
            int lh = static_cast<int>(curves_.size()) * 16 + 10;
            int lw = 104;
            int lx = ml + pw - lw - 4, ly = mt + ph - lh - 6;
            cv::rectangle(canvas, {lx, ly}, {lx+lw, ly+lh}, detail::roc::kPanel, cv::FILLED);
            cv::rectangle(canvas, {lx, ly}, {lx+lw, ly+lh}, detail::roc::kPanel, 1);
            int ci2 = 0;
            for (const auto& [cls, vv] : curves_) {
                cv::Scalar col = detail::roc::kColors[ci2 % static_cast<int>(detail::roc::kColors.size())];
                int ey = ly + 8 + ci2 * 16 + 5;
                cv::line(canvas, {lx+5, ey}, {lx+17, ey}, col, 2, cv::LINE_AA);
                float auc = aucs.count(cls) ? aucs.at(cls) : 0.f;
                auto txt = std::format("{} {:.2f}", cls, auc);
                cv::putText(canvas, txt, {lx+20, ey+4},
                            cv::FONT_HERSHEY_SIMPLEX, 0.28, detail::roc::kText, 1, cv::LINE_AA);
                ++ci2;
            }
        }

        if (!title_.empty())
            cv::putText(canvas, title_, {ml, mt-8},
                        cv::FONT_HERSHEY_SIMPLEX, 0.38, detail::roc::kText, 1, cv::LINE_AA);

        return Image<BGR>(std::move(canvas));
    }

private:
    RocMap      curves_;
    int         width_  = 640;
    int         height_ = 480;
    std::string title_;
};

} // namespace improc::visualization
