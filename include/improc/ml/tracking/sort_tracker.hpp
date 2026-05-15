// include/improc/ml/tracking/sort_tracker.hpp
#pragma once
#include "improc/ml/tracking/track.hpp"
#include <memory>

namespace improc::ml {

struct SortTracker {
    SortTracker& max_age(int frames)    { max_age_  = frames; return *this; }
    SortTracker& min_hits(int hits)     { min_hits_ = hits;   return *this; }
    SortTracker& iou_threshold(float t) { iou_thr_  = t;      return *this; }

    SortTracker();
    ~SortTracker();

    std::vector<Track> update(const std::vector<Detection>& dets);
    void reset();

protected:
    int   max_age_  = 3;
    int   min_hits_ = 3;
    float iou_thr_  = 0.3f;
    int   next_id_  = 0;
    int   frame_    = 0;

    struct KalmanTracklet;
    std::vector<std::unique_ptr<KalmanTracklet>> tracklets_;
};

static_assert(TrackerType<SortTracker>);

} // namespace improc::ml
