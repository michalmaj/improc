// include/improc/ml/eval/segmentation.hpp
#pragma once
#include <map>
#include <stdexcept>
#include <vector>
#include "improc/core/image.hpp"
#include "improc/core/concepts.hpp"

namespace improc::ml {

using improc::core::Gray;
using improc::core::Image;

/// @brief Computes pixel-level IoU for `class_id` over two single-channel label masks.
/// @param pred      Predicted class mask (`Image<Gray>`).
/// @param gt        Ground-truth class mask. Pixels equal to 255 (void label) are ignored.
/// @param class_id  The class index to evaluate.
/// @return IoU in [0, 1], or 0 if neither mask contains `class_id`.
[[nodiscard]] float pixel_iou(const Image<Gray>& pred, const Image<Gray>& gt, int class_id);

/// @brief Computes the Dice coefficient for `class_id` (same void-pixel rule as `pixel_iou`).
/// @param pred      Predicted class mask (`Image<Gray>`).
/// @param gt        Ground-truth class mask. Pixels equal to 255 (void label) are ignored.
/// @param class_id  The class index to evaluate.
/// @return Dice score in [0, 1], or 0 if neither mask contains `class_id`.
[[nodiscard]] float dice(const Image<Gray>& pred, const Image<Gray>& gt, int class_id);

/**
 * @brief Per-class and mean IoU / Dice segmentation metrics.
 */
struct SegMetrics {
    float mIoU      = 0.0f; ///< Mean IoU averaged over all classes present in ground truth.
    float mean_dice = 0.0f; ///< Mean Dice score averaged over all classes present in ground truth.
    std::map<int, float> per_class_iou;  ///< IoU per class index.
    std::map<int, float> per_class_dice; ///< Dice per class index.
};

/**
 * @brief Stateful accumulator for semantic segmentation evaluation.
 *
 * @code
 * SegEval eval;
 * eval.num_classes(21);
 * eval.update(pred_mask, gt_mask);
 * auto metrics = eval.compute();
 * @endcode
 */
struct SegEval {
    /// @brief Sets the number of classes. Resets all accumulated state.
    SegEval& num_classes(int n) {
        num_classes_ = n;
        tp_.assign(n, 0);
        fp_.assign(n, 0);
        fn_.assign(n, 0);
        return *this;
    }

    /// @brief Appends one frame's predicted and ground-truth label masks.
    void update(const Image<Gray>& pred_mask, const Image<Gray>& gt_mask);
    /// @brief Computes and returns segmentation metrics from all accumulated data.
    [[nodiscard]] SegMetrics compute() const;
    /// @brief Resets all accumulated state.
    void reset();

private:
    int num_classes_ = 0;
    std::vector<int64_t> tp_, fp_, fn_;
};

} // namespace improc::ml
