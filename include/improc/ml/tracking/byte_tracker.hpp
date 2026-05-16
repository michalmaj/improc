// include/improc/ml/tracking/byte_tracker.hpp
#pragma once
#include "improc/ml/tracking/track.hpp"
#include "improc/exceptions.hpp"
#include <memory>

namespace improc::ml {

struct ByteTracker {
    ByteTracker& max_age(int frames) {
        if (frames < 0)
            throw improc::ParameterError("max_age", "must be >= 0", "ByteTracker");
        max_age_ = frames; return *this;
    }
    ByteTracker& min_hits(int hits) {
        if (hits < 1)
            throw improc::ParameterError("min_hits", "must be >= 1", "ByteTracker");
        min_hits_ = hits; return *this;
    }
    ByteTracker& high_conf_threshold(float t) {
        if (t < 0.f || t > 1.f)
            throw improc::ParameterError("high_conf_threshold", "must be in [0, 1]", "ByteTracker");
        high_thr_ = t; return *this;
    }
    ByteTracker& low_conf_threshold(float t) {
        if (t < 0.f || t > 1.f)
            throw improc::ParameterError("low_conf_threshold", "must be in [0, 1]", "ByteTracker");
        low_thr_ = t; return *this;
    }

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
