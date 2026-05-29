// include/improc/ml/tracking/track.hpp
#pragma once
#include <vector>
#include "improc/ml/annotated.hpp"
#include "improc/ml/result_types.hpp"

namespace improc::ml {

/**
 * @brief A tracked object instance with a persistent ID across frames.
 *
 * Produced by all tracker `update()` calls. `is_confirmed` is set to `true`
 * once a track has been matched consecutively for `min_hits` frames (SORT /
 * ByteTracker) or after the first match (IouTracker). Unconfirmed tracks are
 * typically filtered out before drawing.
 */
struct Track {
    int   id           = -1;   ///< Unique track ID assigned at birth; -1 = uninitialised.
    BBox  bbox;                ///< Current bounding-box estimate (predicted or measured).
    float confidence   = 0.0f; ///< Detection confidence at the last association.
    int   age          = 0;    ///< Frames elapsed since the track was first created.
    bool  is_confirmed = false; ///< True once the track has been active for `min_hits` frames.
};

/**
 * @brief Ground-truth annotation for a tracked instance in a single frame.
 *
 * Used by `TrackingEval::update()` to compute MOTA, MOTP, and IDF1.
 */
struct TrackGT {
    int  id;   ///< Persistent instance ID across frames (matches across the sequence).
    BBox bbox; ///< Ground-truth bounding box.
};

/**
 * @brief Concept constraining a type to the improc tracker interface.
 *
 * A conforming tracker must expose:
 * - `update(dets) -> std::vector<Track>`: processes one frame of detections and
 *   returns the current set of active tracks.
 * - `reset()`: clears all internal state (tracks, Kalman filters, ID counter).
 */
template<typename T>
concept TrackerType = requires(T tracker, std::vector<Detection> dets) {
    { tracker.update(dets) } -> std::same_as<std::vector<Track>>;
    { tracker.reset()      } -> std::same_as<void>;
};

} // namespace improc::ml
