// include/improc/visualization/pr_curve_plot.hpp
#pragma once

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

namespace detail::pr {
inline const cv::Scalar kBg    {42,  23,  15};
inline const cv::Scalar kPanel {59,  41,  30};
inline const cv::Scalar kText  {184, 163, 148};
inline const cv::Scalar kMuted {139, 116, 100};
inline const cv::Scalar kAccent{250, 139, 167};
inline const std::vector<cv::Scalar> kColors{
    {250, 139, 167}, {248, 189,  56}, {153, 211,  52}, { 11, 158, 245}
};
} // namespace detail::pr

struct PRCurvePlot {
    using CurveMap = std::map<std::string,
        std::pair<std::vector<float>, std::vector<float>>>;

    // Struct form — direct from DetectionEval::pr_curves()
    explicit PRCurvePlot(CurveMap curves)
        : curves_(std::move(curves)) {}

    // Raw form — separate recall and precision maps
    PRCurvePlot(std::map<std::string, std::vector<float>> recalls,
                std::map<std::string, std::vector<float>> precisions) {
        for (const auto& [cls, rec] : recalls) {
            auto it = precisions.find(cls);
            if (it != precisions.end())
                curves_[cls] = {rec, it->second};
        }
    }

    PRCurvePlot& width(int w)           { width_ = w;            return *this; }
    PRCurvePlot& height(int h)          { height_ = h;           return *this; }
    PRCurvePlot& title(std::string t)   { title_ = std::move(t); return *this; }
    PRCurvePlot& mAP_50(float v)        { mAP50_ = v;            return *this; }
    PRCurvePlot& iou_threshold(float t) { iou_thr_ = t;          return *this; }

    Image<BGR> operator()() const {
        if (width_ <= 0)  throw improc::ParameterError{"width",  "must be positive", "PRCurvePlot"};
        if (height_ <= 0) throw improc::ParameterError{"height", "must be positive", "PRCurvePlot"};

        cv::Mat canvas(height_, width_, CV_8UC3, detail::pr::kBg);

        const int ml = 48, mr = 12, mt = 28, mb = 48;
        const int pw = width_ - ml - mr;
        const int ph = height_ - mt - mb;

        // Axes
        cv::line(canvas, {ml, mt},      {ml, mt+ph},     detail::pr::kPanel, 1);
        cv::line(canvas, {ml, mt+ph},   {ml+pw, mt+ph},  detail::pr::kPanel, 1);

        // Dashed grid at y=0.5
        for (int x = ml; x < ml+pw; x += 8)
            cv::line(canvas, {x, mt+ph/2}, {std::min(x+4, ml+pw), mt+ph/2},
                     detail::pr::kPanel, 1);

        // Y axis labels
        auto yl = [&](float v, int y) {
            cv::putText(canvas, std::format("{:.1f}", v), {ml-38, y+4},
                        cv::FONT_HERSHEY_SIMPLEX, 0.28, detail::pr::kText, 1, cv::LINE_AA);
        };
        yl(1.0f, mt);
        yl(0.75f, mt + ph/4);
        yl(0.5f,  mt + ph/2);
        yl(0.25f, mt + 3*ph/4);
        yl(0.0f,  mt + ph);

        // Coordinate mapping
        auto sx = [&](float r) { return ml + static_cast<int>(r * pw); };
        auto sy = [&](float p) { return mt + ph - static_cast<int>(p * ph); };

        // Curves
        int ci = 0;
        for (const auto& [cls, vecs] : curves_) {
            const auto& [recs, precs] = vecs;
            if (recs.empty()) { ++ci; continue; }
            cv::Scalar col = detail::pr::kColors[ci % static_cast<int>(detail::pr::kColors.size())];
            std::vector<cv::Point> pts;
            pts.reserve(recs.size());
            for (std::size_t i = 0; i < recs.size(); ++i)
                pts.push_back({sx(recs[i]), sy(precs[i])});
            cv::polylines(canvas, pts, false, col, 2, cv::LINE_AA);
            ++ci;
        }

        // Legend box (top-right)
        if (!curves_.empty()) {
            int lh = static_cast<int>(curves_.size()) * 16 + 10;
            int lw = 80;
            int lx = ml + pw - lw - 4, ly = mt + 6;
            cv::rectangle(canvas, {lx, ly}, {lx+lw, ly+lh}, detail::pr::kPanel, cv::FILLED);
            cv::rectangle(canvas, {lx, ly}, {lx+lw, ly+lh}, detail::pr::kPanel, 1);
            int ci2 = 0;
            for (const auto& [cls, vv] : curves_) {
                cv::Scalar col = detail::pr::kColors[ci2 % static_cast<int>(detail::pr::kColors.size())];
                int ey = ly + 8 + ci2 * 16 + 5;
                cv::line(canvas, {lx+5, ey}, {lx+17, ey}, col, 2, cv::LINE_AA);
                cv::putText(canvas, cls, {lx+20, ey+4},
                            cv::FONT_HERSHEY_SIMPLEX, 0.28, detail::pr::kText, 1, cv::LINE_AA);
                ++ci2;
            }
        }

        // X axis label
        cv::putText(canvas, "Recall",
                    {ml + pw/2 - 18, mt+ph+32},
                    cv::FONT_HERSHEY_SIMPLEX, 0.32, detail::pr::kMuted, 1, cv::LINE_AA);

        // mAP badge (bottom)
        if (mAP50_ >= 0.0f) {
            int bx = ml, by = height_ - 18;
            cv::rectangle(canvas, {bx, by-12}, {bx+pw, by+4}, detail::pr::kPanel, cv::FILLED);
            auto txt = std::format("mAP@{:.2f}: {:.3f}", iou_thr_, mAP50_);
            cv::putText(canvas, txt, {bx+8, by},
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, detail::pr::kAccent, 1, cv::LINE_AA);
        }

        if (!title_.empty())
            cv::putText(canvas, title_, {ml, mt-8},
                        cv::FONT_HERSHEY_SIMPLEX, 0.38, detail::pr::kText, 1, cv::LINE_AA);

        return Image<BGR>(std::move(canvas));
    }

private:
    CurveMap    curves_;
    int         width_   = 640;
    int         height_  = 480;
    float       mAP50_   = -1.0f;  // <0 means don't show badge
    float       iou_thr_ = 0.50f;
    std::string title_;
};

} // namespace improc::visualization
