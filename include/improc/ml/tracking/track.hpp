// include/improc/ml/tracking/track.hpp
#pragma once
#include <vector>
#include "improc/ml/annotated.hpp"
#include "improc/ml/result_types.hpp"

namespace improc::ml {

struct Track {
    int   id           = -1;
    BBox  bbox;
    float confidence   = 0.0f;
    int   age          = 0;
    bool  is_confirmed = false;
};

/// Ground truth for tracking: bbox + persistent instance ID across frames.
struct TrackGT {
    int  id;
    BBox bbox;
};

template<typename T>
concept TrackerType = requires(T tracker, std::vector<Detection> dets) {
    { tracker.update(dets) } -> std::same_as<std::vector<Track>>;
    { tracker.reset()      } -> std::same_as<void>;
};

} // namespace improc::ml
