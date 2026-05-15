// include/improc/ml/eval/detection.hpp
#pragma once
#include <array>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include "improc/ml/annotated.hpp"
#include "improc/ml/result_types.hpp"

namespace improc::ml {

// Intersection over Union of two bounding boxes.
// Returns 0.0 when either box has zero area.
float iou(const BBox& a, const BBox& b);

// Area under the precision-recall curve using 101-point COCO interpolation.
// recalls and precisions must have the same length.
float average_precision(std::span<const float> recalls,
                        std::span<const float> precisions);

struct DetectionMetrics {
    float mAP_50    = 0.0f;
    float mAP_50_95 = 0.0f;
    std::map<std::string, float> per_class_AP;
};

struct DetectionEval {
    void update(const std::vector<Detection>& preds,
                const std::vector<BBox>&      gts);
    DetectionMetrics compute() const;
    void reset();

private:
    static constexpr std::array<float, 10> kThresholds_{
        0.50f, 0.55f, 0.60f, 0.65f, 0.70f,
        0.75f, 0.80f, 0.85f, 0.90f, 0.95f
    };
    std::array<std::map<std::string, std::vector<std::pair<float,bool>>>, 10> matches_;
    std::map<std::string, int> gt_counts_;
};

} // namespace improc::ml
