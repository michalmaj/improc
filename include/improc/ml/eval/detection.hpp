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

/// @brief Computes the Intersection-over-Union of two axis-aligned bounding boxes.
/// @param a First bounding box.
/// @param b Second bounding box.
/// @return Value in [0, 1]. Returns 0 if either box has zero area.
[[nodiscard]] float iou(const BBox& a, const BBox& b);

/// @brief Computes Average Precision (AP) using the 101-point interpolation (COCO style).
/// @param recalls    Recall values in ascending order.
/// @param precisions Precision values corresponding to `recalls`.
/// @return AP in [0, 1].
[[nodiscard]] float average_precision(std::span<const float> recalls,
                                      std::span<const float> precisions);

/**
 * @brief mAP and per-class AP detection metrics at IoU thresholds 0.50 and 0.50:0.95.
 */
struct DetectionMetrics {
    float mAP_50    = 0.0f; ///< Mean Average Precision at IoU ≥ 0.50.
    float mAP_50_95 = 0.0f; ///< Mean AP averaged over IoU thresholds 0.50–0.95 (step 0.05).
    std::map<std::string, float> per_class_AP; ///< AP@0.50 per class name.
};

/**
 * @brief Stateful accumulator for object detection evaluation.
 *
 * Accumulates predictions and ground truths across frames, then computes
 * mAP@0.50 and mAP@0.50:0.95 matching COCO evaluation conventions.
 *
 * @code
 * DetectionEval eval;
 * eval.update(predictions, ground_truths);
 * auto metrics = eval.compute();
 * @endcode
 */
struct DetectionEval {
    /// @brief Appends a frame's predicted `Detection` list and ground-truth `BBox` list.
    void update(const std::vector<Detection>& preds,
                const std::vector<BBox>&      gts);
    /// @brief Computes and returns detection metrics from all accumulated data.
    [[nodiscard]] DetectionMetrics compute() const;
    /// @brief Returns recall and precision curves per class at IoU=0.50.
    ///
    /// Recalls are non-decreasing. Returns empty map before any `update()` calls.
    [[nodiscard]]
    std::map<std::string, std::pair<std::vector<float>, std::vector<float>>>
    pr_curves() const;
    /// @brief Resets all accumulated state.
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
