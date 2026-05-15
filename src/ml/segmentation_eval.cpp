// src/ml/segmentation_eval.cpp
#include "improc/ml/eval/segmentation.hpp"
#include <stdexcept>

namespace improc::ml {

float pixel_iou(const Image<Gray>& pred, const Image<Gray>& gt, int class_id) {
    int64_t tp = 0, fp = 0, fn = 0;
    const auto& pm = pred.mat();
    const auto& gm = gt.mat();
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
    int64_t tp = 0, fp = 0, fn = 0;
    const auto& pm = pred.mat();
    const auto& gm = gt.mat();
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

void SegEval::update(const Image<Gray>&, const Image<Gray>&) {}
SegMetrics SegEval::compute() const { return {}; }
void SegEval::reset() { tp_.assign(num_classes_, 0); fp_.assign(num_classes_, 0); fn_.assign(num_classes_, 0); }

} // namespace improc::ml
