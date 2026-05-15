// include/improc/ml/tracking/byte_tracker.hpp
#pragma once
#include "improc/ml/tracking/track.hpp"
#include <memory>

namespace improc::ml {

struct ByteTracker {
    ByteTracker& max_age(int frames)          { max_age_  = frames; return *this; }
    ByteTracker& min_hits(int hits)           { min_hits_ = hits;   return *this; }
    ByteTracker& high_conf_threshold(float t) { high_thr_ = t;      return *this; }
    ByteTracker& low_conf_threshold(float t)  { low_thr_  = t;      return *this; }

    [[nodiscard]] std::vector<Track> update(const std::vector<Detection>& dets);
    void reset();

    ~ByteTracker();
    ByteTracker();

private:
    int   max_age_  = 3;
    int   min_hits_ = 3;
    float high_thr_ = 0.6f;
    float low_thr_  = 0.1f;
    int   next_id_  = 0;

    struct KalmanTracklet;
    std::vector<std::unique_ptr<KalmanTracklet>> tracklets_;
};

static_assert(TrackerType<ByteTracker>);

} // namespace improc::ml
