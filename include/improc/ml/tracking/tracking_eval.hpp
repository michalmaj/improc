// include/improc/ml/tracking/tracking_eval.hpp
#pragma once
#include <map>
#include <utility>
#include <vector>
#include "improc/ml/tracking/track.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

/**
 * @brief Aggregated multi-object tracking metrics for a complete sequence.
 *
 * Computed by `TrackingEval::compute()` after accumulating frames with `update()`.
 * All metrics follow the MOTChallenge definitions.
 */
struct TrackingMetrics {
    float MOTA      = 0.0f; ///< Multiple Object Tracking Accuracy: `1 - (FN + FP + IDSW) / GT_total`. Range (-∞, 1].
    float MOTP      = 0.0f; ///< Multiple Object Tracking Precision: mean IoU of matched track–GT pairs. Range [0, 1].
    float IDF1      = 0.0f; ///< ID F1 score: `2·IDTP / (2·IDTP + IDFP + IDFN)`. Measures ID consistency. Range [0, 1].
    float Precision = 0.0f; ///< Detection precision: `TP / (TP + FP)`. Range [0, 1].
    float Recall    = 0.0f; ///< Detection recall: `TP / (TP + FN)`. Range [0, 1].
    int   FP        = 0;    ///< Total false-positive detections across all frames.
    int   FN        = 0;    ///< Total false-negative (missed) detections across all frames.
    int   IDSW      = 0;    ///< Total identity switches: a GT object swapped its matched track ID.
};

/**
 * @brief Stateful accumulator that computes MOT metrics over a detection sequence.
 *
 * Call `update()` once per frame with predicted tracks and ground-truth annotations,
 * then call `compute()` at the end of the sequence to obtain `TrackingMetrics`.
 * Call `reset()` to reuse the evaluator on a new sequence.
 *
 * @code
 * TrackingEval eval;
 * eval.iou_threshold(0.5f);
 * for (auto& [tracks, gts] : sequence)
 *     eval.update(tracks, gts);
 * TrackingMetrics m = eval.compute();
 * std::cout << "MOTA=" << m.MOTA << "  IDF1=" << m.IDF1 << "\n";
 * @endcode
 */
struct TrackingEval {
    /// @brief Sets the minimum IoU overlap to count a track–GT pair as a true positive (default: 0.5).
    /// @throws improc::ParameterError if `t` is outside [0, 1].
    TrackingEval& iou_threshold(float t) {
        if (t < 0.f || t > 1.f)
            throw improc::ParameterError("iou_threshold", "must be in [0, 1]", "TrackingEval");
        iou_thr_ = t; return *this;
    }

    /// @brief Accumulates one frame of predicted tracks against ground-truth annotations.
    void update(const std::vector<Track>&   tracks,
                const std::vector<TrackGT>& gts);
    /// @brief Computes and returns aggregate metrics over all accumulated frames.
    [[nodiscard]] TrackingMetrics compute() const;
    /// @brief Resets all accumulated state; the evaluator can then be reused for a new sequence.
    void reset();

private:
    float iou_thr_     = 0.5f;
    int   FP_          = 0;
    int   FN_          = 0;
    int   IDSW_        = 0;
    int   GT_total_    = 0;
    float iou_sum_     = 0.0f;
    int   match_count_ = 0;

    std::map<int, int>                 last_track_for_gt_;  // gt_id -> last matched track_id
    std::map<int, int>                 track_total_;        // track_id -> frames active
    std::map<int, int>                 gt_total_;           // gt_id -> frames present
    std::map<std::pair<int,int>, int>  pair_frames_;        // (track_id, gt_id) -> co-matched frames
};

} // namespace improc::ml
