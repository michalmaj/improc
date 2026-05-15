// include/improc/ml/tracking/iou_tracker.hpp
#pragma once
#include "improc/ml/tracking/track.hpp"

namespace improc::ml {

struct IouTracker {
    IouTracker& min_iou(float t)    { min_iou_ = t; return *this; }
    IouTracker& max_age(int frames) { max_age_ = frames; return *this; }

    std::vector<Track> update(const std::vector<Detection>& dets);
    void reset();

private:
    float min_iou_ = 0.3f;
    int   max_age_ = 1;
    int   next_id_ = 0;

    struct InternalTrack {
        Track track;
        BBox  bbox;
    };
    std::vector<InternalTrack> tracks_;
};

static_assert(TrackerType<IouTracker>);

} // namespace improc::ml
