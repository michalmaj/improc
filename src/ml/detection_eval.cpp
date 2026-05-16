// src/ml/detection_eval.cpp
#include "improc/ml/eval/detection.hpp"
#include <algorithm>
#include <set>

namespace improc::ml {

namespace {

std::string class_key(const std::string& label, int class_id) {
    return label.empty() ? std::to_string(class_id) : label;
}

float ap_for_class(const std::vector<std::pair<float,bool>>& matches, int gt_count) {
    if (gt_count == 0 || matches.empty()) return 0.0f;
    auto sorted = matches;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<float> rec, prec;
    rec.reserve(sorted.size());
    prec.reserve(sorted.size());
    int tp = 0, fp = 0;
    for (const auto& [conf, ok] : sorted) {
        if (ok) ++tp; else ++fp;
        rec.push_back(static_cast<float>(tp) / gt_count);
        prec.push_back(static_cast<float>(tp) / (tp + fp));
    }
    return average_precision(rec, prec);
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

void DetectionEval::update(const std::vector<Detection>& preds,
                            const std::vector<BBox>& gts) {
    for (const auto& gt : gts)
        gt_counts_[class_key(gt.label, gt.class_id)]++;

    auto sorted = preds;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  return a.confidence > b.confidence;
              });

    for (int ti = 0; ti < 10; ++ti) {
        float thr = kThresholds_[ti];
        std::vector<bool> used(gts.size(), false);

        for (const auto& pred : sorted) {
            std::string key = class_key(pred.label, pred.class_id);
            BBox pred_box{pred.box, pred.class_id, pred.label};

            float best_iou = 0.0f;
            int   best_j   = -1;
            for (std::size_t j = 0; j < gts.size(); ++j) {
                if (used[j]) continue;
                if (class_key(gts[j].label, gts[j].class_id) != key) continue;
                float s = iou(pred_box, gts[j]);
                if (s > best_iou) { best_iou = s; best_j = static_cast<int>(j); }
            }
            bool is_tp = best_j >= 0 && best_iou >= thr;
            if (is_tp) used[best_j] = true;
            matches_[ti][key].emplace_back(pred.confidence, is_tp);
        }
    }
}

DetectionMetrics DetectionEval::compute() const {
    std::set<std::string> classes;
    for (const auto& [cls, _] : gt_counts_) classes.insert(cls);
    for (const auto& tm : matches_)
        for (const auto& [cls, _] : tm) classes.insert(cls);

    if (classes.empty()) return {};

    std::map<std::string, std::array<float, 10>> class_ap;
    for (const auto& cls : classes) {
        int total = gt_counts_.count(cls) ? gt_counts_.at(cls) : 0;
        for (int ti = 0; ti < 10; ++ti) {
            auto it = matches_[ti].find(cls);
            class_ap[cls][ti] = (it == matches_[ti].end())
                ? 0.0f
                : ap_for_class(it->second, total);
        }
    }

    DetectionMetrics out;
    float sum_50 = 0.0f, sum_5095 = 0.0f;
    for (const auto& [cls, aps] : class_ap) {
        out.per_class_AP[cls] = aps[0];
        sum_50 += aps[0];
        float cls_mean = 0.0f;
        for (float ap : aps) cls_mean += ap;
        sum_5095 += cls_mean / 10.0f;
    }
    float n = static_cast<float>(classes.size());
    out.mAP_50    = sum_50   / n;
    out.mAP_50_95 = sum_5095 / n;
    return out;
}

std::map<std::string, std::pair<std::vector<float>, std::vector<float>>>
DetectionEval::pr_curves() const {
    std::map<std::string, std::pair<std::vector<float>, std::vector<float>>> result;
    // Use IoU=0.50 (index 0 in kThresholds_)
    for (const auto& [cls, matches] : matches_[0]) {
        int gt_count = gt_counts_.count(cls) ? gt_counts_.at(cls) : 0;
        if (gt_count == 0 || matches.empty()) continue;
        auto sorted = matches;
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<float> rec, prec;
        rec.reserve(sorted.size());
        prec.reserve(sorted.size());
        int tp = 0, fp = 0;
        for (const auto& [conf, ok] : sorted) {
            if (ok) ++tp; else ++fp;
            rec.push_back(static_cast<float>(tp) / gt_count);
            prec.push_back(static_cast<float>(tp) / (tp + fp));
        }
        result[cls] = {std::move(rec), std::move(prec)};
    }
    return result;
}

void DetectionEval::reset() {
    for (auto& m : matches_) m.clear();
    gt_counts_.clear();
}

} // namespace improc::ml
