// include/improc/visualization/confusion_matrix_plot.hpp
#pragma once

#include <format>
#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "improc/core/image.hpp"
#include "improc/exceptions.hpp"
#include "improc/ml/eval/classification.hpp"

namespace improc::visualization {

using improc::core::Image;
using improc::core::BGR;

namespace {
const cv::Scalar kCmpBg    {42,  23,  15};
const cv::Scalar kCmpPanel {59,  41,  30};
const cv::Scalar kCmpText  {184, 163, 148};
const cv::Scalar kCmpMuted {139, 116, 100};
const cv::Scalar kCmpAccent{250, 139, 167};
const cv::Scalar kCmpAmber { 11, 158, 245};
const cv::Scalar kCmpRed   { 68,  68, 239};
} // namespace

struct ConfusionMatrixPlot {
    // Struct form from ConfusionMatrix
    explicit ConfusionMatrixPlot(const improc::ml::ConfusionMatrix& cm)
        : num_classes_(cm.num_classes) {
        if (!cm.mat.empty()) {
            int n = cm.mat.rows;
            matrix_.resize(n, std::vector<int>(n));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    matrix_[i][j] = cm.mat(i, j);
        }
    }

    // Struct form from ClassMetrics (uses embedded confusion matrix)
    explicit ConfusionMatrixPlot(const improc::ml::ClassMetrics& m)
        : ConfusionMatrixPlot(m.confusion_matrix) {}

    // Raw form: 2D vector + optional class names
    explicit ConfusionMatrixPlot(std::vector<std::vector<int>> matrix,
                                 std::vector<std::string> names = {})
        : matrix_(std::move(matrix))
        , class_names_(std::move(names))
        , num_classes_(static_cast<int>(matrix_.size())) {}

    ConfusionMatrixPlot& width(int w)          { width_ = w;              return *this; }
    ConfusionMatrixPlot& height(int h)         { height_ = h;             return *this; }
    ConfusionMatrixPlot& title(std::string t)  { title_ = std::move(t);   return *this; }
    ConfusionMatrixPlot& class_names(std::vector<std::string> n) {
        class_names_ = std::move(n); return *this;
    }

    Image<BGR> operator()() const {
        if (width_ <= 0)  throw improc::ParameterError{"width",  "must be positive", "ConfusionMatrixPlot"};
        if (height_ <= 0) throw improc::ParameterError{"height", "must be positive", "ConfusionMatrixPlot"};

        cv::Mat canvas(height_, width_, CV_8UC3, kCmpBg);
        if (matrix_.empty()) return Image<BGR>(std::move(canvas));

        int n = static_cast<int>(matrix_.size());
        const int ml = 62, mr = 28, mt = 38, mb = 42;
        const int pw = width_ - ml - mr;
        const int ph = height_ - mt - mb;
        const int cw = pw / n, ch = ph / n;

        // Find max value for color scaling
        int max_val = 1;
        for (const auto& row : matrix_)
            for (int v : row) max_val = std::max(max_val, v);

        auto diag_col = [&](int val) -> cv::Scalar {
            float t = std::min(1.0f, static_cast<float>(val) / max_val);
            return cv::Scalar(42 + (237-42)*t, 23 + (58-23)*t, 15 + (124-15)*t);
        };
        auto off_col = [&](int val) -> cv::Scalar {
            if (val == 0) return kCmpBg;
            float t = std::min(1.0f, static_cast<float>(val) / max_val * 4.0f);
            if (t < 0.5f) {
                float s = t * 2.0f;
                return cv::Scalar(42*(1-s) + 11*s, 23*(1-s) + 158*s, 15*(1-s) + 245*s);
            }
            return kCmpRed;
        };

        // Draw cells
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                int val = matrix_[r][c];
                cv::Rect cell(ml + c*cw, mt + r*ch, cw-1, ch-1);
                cv::Scalar col = (r == c) ? diag_col(val) : off_col(val);
                cv::rectangle(canvas, cell, col, cv::FILLED);
                // Value text centered
                std::string s = std::to_string(val);
                int bl = 0;
                auto sz = cv::getTextSize(s, cv::FONT_HERSHEY_SIMPLEX, 0.38, 1, &bl);
                cv::Point tp(cell.x + (cw - sz.width)/2,
                             cell.y + (ch + sz.height)/2);
                cv::Scalar tc = (static_cast<float>(val) / max_val > 0.35f)
                    ? cv::Scalar(255,255,255) : kCmpText;
                cv::putText(canvas, s, tp, cv::FONT_HERSHEY_SIMPLEX, 0.38, tc, 1, cv::LINE_AA);
            }
        }

        // Column headers
        cv::putText(canvas, "Predicted",
                    {ml + pw/2 - 32, mt - 20},
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, kCmpAccent, 1, cv::LINE_AA);
        for (int c = 0; c < n; ++c) {
            std::string nm = c < static_cast<int>(class_names_.size())
                ? class_names_[c] : std::to_string(c);
            cv::putText(canvas, nm, {ml + c*cw + 3, mt - 6},
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, kCmpText, 1, cv::LINE_AA);
        }

        // Row headers + "True" label
        cv::putText(canvas, "True",
                    {4, mt + ph/2 + 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, kCmpAccent, 1, cv::LINE_AA);
        for (int r = 0; r < n; ++r) {
            std::string nm = r < static_cast<int>(class_names_.size())
                ? class_names_[r] : std::to_string(r);
            cv::putText(canvas, nm,
                        {4, mt + r*ch + ch/2 + 4},
                        cv::FONT_HERSHEY_SIMPLEX, 0.28, kCmpText, 1, cv::LINE_AA);
        }

        // Colorbar (right margin)
        int cbx = width_ - mr + 4;
        for (int y = 0; y < ph; ++y) {
            float t = 1.0f - static_cast<float>(y) / ph;
            cv::Scalar col(42 + (237-42)*t, 23 + (58-23)*t, 15 + (124-15)*t);
            cv::line(canvas, {cbx, mt+y}, {cbx+10, mt+y}, col, 1);
        }
        cv::putText(canvas, std::to_string(max_val),
                    {cbx, mt - 2}, cv::FONT_HERSHEY_SIMPLEX, 0.26, kCmpMuted, 1);
        cv::putText(canvas, "0",
                    {cbx, mt+ph+8}, cv::FONT_HERSHEY_SIMPLEX, 0.26, kCmpMuted, 1);

        // Accuracy + Macro F1 badge
        {
            int total = 0, correct = 0;
            for (int i = 0; i < n; ++i) {
                correct += matrix_[i][i];
                for (int j = 0; j < n; ++j) total += matrix_[i][j];
            }
            float acc = total > 0 ? static_cast<float>(correct) / total : 0.f;
            float f1_sum = 0.f;
            for (int c = 0; c < n; ++c) {
                int tp = matrix_[c][c], col_s = 0, row_s = 0;
                for (int i = 0; i < n; ++i) col_s += matrix_[i][c];
                for (int j = 0; j < n; ++j) row_s += matrix_[c][j];
                float p = col_s > 0 ? static_cast<float>(tp)/col_s : 0.f;
                float r = row_s > 0 ? static_cast<float>(tp)/row_s : 0.f;
                f1_sum += (p+r) > 0.f ? 2*p*r/(p+r) : 0.f;
            }
            float mf1 = n > 0 ? f1_sum / n : 0.f;
            int by = height_ - mb + 8;
            cv::rectangle(canvas, {ml, by}, {ml+pw, by+26}, kCmpPanel, cv::FILLED);
            auto txt = std::format("Accuracy: {:.1f}%  |  Macro F1: {:.2f}",
                                   acc * 100.f, mf1);
            cv::putText(canvas, txt, {ml + 8, by + 16},
                        cv::FONT_HERSHEY_SIMPLEX, 0.34, kCmpAccent, 1, cv::LINE_AA);
        }

        if (!title_.empty())
            cv::putText(canvas, title_, {ml, 18},
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, kCmpText, 1, cv::LINE_AA);

        return Image<BGR>(std::move(canvas));
    }

private:
    std::vector<std::vector<int>> matrix_;
    std::vector<std::string>      class_names_;
    int    num_classes_ = 0;
    int    width_  = 400;
    int    height_ = 400;
    std::string title_;
};

} // namespace improc::visualization
