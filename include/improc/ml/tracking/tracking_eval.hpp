// include/improc/ml/tracking/tracking_eval.hpp
#pragma once
#include <map>
#include <utility>
#include <vector>
#include "improc/ml/tracking/track.hpp"

namespace improc::ml {

struct TrackingMetrics {
    float MOTA = 0.0f;  // 1 - (FN + FP + IDSW) / GT_total
    float MOTP = 0.0f;  // mean IoU of matched pairs
    float IDF1 = 0.0f;  // 2*IDTP / (2*IDTP + IDFP + IDFN)
    int   FP   = 0;
    int   FN   = 0;
    int   IDSW = 0;
};

struct TrackingEval {
    TrackingEval& iou_threshold(float t) { iou_thr_ = t; return *this; }

    void update(const std::vector<Track>&   tracks,
                const std::vector<TrackGT>& gts);
    [[nodiscard]] TrackingMetrics compute() const;
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
