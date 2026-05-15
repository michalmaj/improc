// src/ml/detection_eval.cpp
#include "improc/ml/eval/detection.hpp"
#include <algorithm>
#include <set>

namespace improc::ml {

namespace {
std::string class_key(const std::string& label, int class_id) {
    return label.empty() ? std::to_string(class_id) : label;
}
} // namespace

float iou(const BBox& a, const BBox& b) {
    float ix1 = std::max(a.box.x, b.box.x);
    float iy1 = std::max(a.box.y, b.box.y);
    float ix2 = std::min(a.box.x + a.box.width,  b.box.x + b.box.width);
    float iy2 = std::min(a.box.y + a.box.height, b.box.y + b.box.height);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float area_a = a.box.width * a.box.height;
    float area_b = b.box.width * b.box.height;
    float uni = area_a + area_b - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}

float average_precision(std::span<const float> recalls,
                        std::span<const float> precisions) {
    if (recalls.size() != precisions.size() || recalls.empty()) return 0.0f;
    float ap = 0.0f;
    for (int i = 0; i <= 100; ++i) {
        float thr = static_cast<float>(i) / 100.0f;
        float p = 0.0f;
        for (std::size_t j = 0; j < recalls.size(); ++j)
            if (recalls[j] >= thr) p = std::max(p, precisions[j]);
        ap += p;
    }
    return ap / 101.0f;
}

void DetectionEval::update(const std::vector<Detection>&, const std::vector<BBox>&) {}
DetectionMetrics DetectionEval::compute() const { return {}; }
void DetectionEval::reset() {}

} // namespace improc::ml
