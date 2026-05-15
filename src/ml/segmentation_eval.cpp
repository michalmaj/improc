// src/ml/segmentation_eval.cpp
#include "improc/ml/eval/segmentation.hpp"
#include <stdexcept>

namespace improc::ml {

float pixel_iou(const Image<Gray>& pred, const Image<Gray>& gt, int class_id) {
    const auto& pm = pred.mat();
    const auto& gm = gt.mat();
    if (pm.rows != gm.rows || pm.cols != gm.cols)
        throw std::invalid_argument("pixel_iou: pred and gt dimensions must match");
    int64_t tp = 0, fp = 0, fn = 0;
    for (int r = 0; r < pm.rows; ++r) {
        for (int c = 0; c < pm.cols; ++c) {
            uint8_t p_val = pm.at<uint8_t>(r, c);
            uint8_t g_val = gm.at<uint8_t>(r, c);
            if (g_val == 255) continue;
            bool p_c = (p_val == static_cast<uint8_t>(class_id));
            bool g_c = (g_val == static_cast<uint8_t>(class_id));
            if (p_c && g_c)       ++tp;
            else if (p_c && !g_c) ++fp;
            else if (!p_c && g_c) ++fn;
        }
    }
    int64_t denom = tp + fp + fn;
    return denom == 0 ? 0.0f : static_cast<float>(tp) / static_cast<float>(denom);
}

float dice(const Image<Gray>& pred, const Image<Gray>& gt, int class_id) {
    const auto& pm = pred.mat();
    const auto& gm = gt.mat();
    if (pm.rows != gm.rows || pm.cols != gm.cols)
        throw std::invalid_argument("dice: pred and gt dimensions must match");
    int64_t tp = 0, fp = 0, fn = 0;
    for (int r = 0; r < pm.rows; ++r) {
        for (int c = 0; c < pm.cols; ++c) {
            uint8_t p_val = pm.at<uint8_t>(r, c);
            uint8_t g_val = gm.at<uint8_t>(r, c);
            if (g_val == 255) continue;
            bool p_c = (p_val == static_cast<uint8_t>(class_id));
            bool g_c = (g_val == static_cast<uint8_t>(class_id));
            if (p_c && g_c)       ++tp;
            else if (p_c && !g_c) ++fp;
            else if (!p_c && g_c) ++fn;
        }
    }
    int64_t denom = 2*tp + fp + fn;
    return denom == 0 ? 0.0f : static_cast<float>(2*tp) / static_cast<float>(denom);
}

void SegEval::update(const Image<Gray>& pred, const Image<Gray>& gt) {
    if (num_classes_ <= 0)
        throw std::runtime_error("SegEval: call num_classes() before update()");
    if (pred.mat().rows != gt.mat().rows || pred.mat().cols != gt.mat().cols)
        throw std::invalid_argument("SegEval: pred and gt dimensions must match");
    const auto& pm = pred.mat();
    const auto& gm = gt.mat();
    for (int r = 0; r < pm.rows; ++r) {
        for (int c = 0; c < pm.cols; ++c) {
            int p = pm.at<uint8_t>(r, c);
            int g = gm.at<uint8_t>(r, c);
            if (g == 255) continue;
            if (g >= num_classes_) continue;
            if (p == g) {
                ++tp_[g];
            } else {
                ++fn_[g];
                if (p < num_classes_) ++fp_[p];
            }
        }
    }
}

SegMetrics SegEval::compute() const {
    SegMetrics m;
    float sum_iou = 0.0f, sum_dice = 0.0f;
    int count = 0;
    for (int c = 0; c < num_classes_; ++c) {
        int64_t tp = tp_[c], fp = fp_[c], fn = fn_[c];
        if (tp + fp + fn == 0) continue;
        float iou_c  = static_cast<float>(tp) / static_cast<float>(tp + fp + fn);
        int64_t dd   = 2*tp + fp + fn;
        float dice_c = dd == 0 ? 0.0f : static_cast<float>(2*tp) / static_cast<float>(dd);
        m.per_class_iou[c]  = iou_c;
        m.per_class_dice[c] = dice_c;
        sum_iou  += iou_c;
        sum_dice += dice_c;
        ++count;
    }
    m.mIoU      = count > 0 ? sum_iou  / count : 0.0f;
    m.mean_dice = count > 0 ? sum_dice / count : 0.0f;
    return m;
}

void SegEval::reset() { tp_.assign(num_classes_, 0); fp_.assign(num_classes_, 0); fn_.assign(num_classes_, 0); }

} // namespace improc::ml
