// include/improc/ml/tracking/sort_tracker.hpp
#pragma once
#include "improc/ml/tracking/track.hpp"
#include "improc/exceptions.hpp"
#include <memory>

namespace improc::ml {

/**
 * @brief SORT: Simple Online and Realtime Tracking — Kalman filter + Hungarian algorithm.
 *
 * Each track maintains a constant-velocity Kalman filter that predicts the next
 * bounding box. Incoming detections are assigned to predicted positions via the
 * Hungarian algorithm using IoU as the cost metric. Tracks are promoted to
 * "confirmed" after `min_hits` consecutive matches, and deleted after `max_age`
 * unmatched frames.
 *
 * Reference: Bewley et al., "Simple Online and Realtime Tracking", ICASSP 2016.
 *
 * @code
 * SortTracker tracker;
 * tracker.max_age(3).min_hits(3).iou_threshold(0.3f);
 * std::vector<Track> tracks = tracker.update(detections);
 * @endcode
 */
struct SortTracker {
    /// @brief Sets the number of consecutive unmatched frames before a track is deleted (default: 3).
    /// @throws improc::ParameterError if `frames` < 0.
    SortTracker& max_age(int frames) {
        if (frames < 0)
            throw improc::ParameterError("max_age", "must be >= 0", "SortTracker");
        max_age_ = frames; return *this;
    }
    /// @brief Sets the minimum consecutive matches required before a track is confirmed (default: 3).
    /// @throws improc::ParameterError if `hits` < 1.
    SortTracker& min_hits(int hits) {
        if (hits < 1)
            throw improc::ParameterError("min_hits", "must be >= 1", "SortTracker");
        min_hits_ = hits; return *this;
    }
    /// @brief Sets the minimum IoU overlap for the Hungarian assignment step (default: 0.3).
    /// @throws improc::ParameterError if `t` is outside [0, 1].
    SortTracker& iou_threshold(float t) {
        if (t < 0.f || t > 1.f)
            throw improc::ParameterError("iou_threshold", "must be in [0, 1]", "SortTracker");
        iou_thr_ = t; return *this;
    }

    SortTracker();
    ~SortTracker();

    /// @brief Processes one frame of detections and returns currently active tracks.
    [[nodiscard]] std::vector<Track> update(const std::vector<Detection>& dets);
    /// @brief Clears all tracks and resets the ID counter.
    void reset();

private:
    int   max_age_  = 3;
    int   min_hits_ = 3;
    float iou_thr_  = 0.3f;
    int   next_id_  = 0;

    struct KalmanTracklet;
    std::vector<std::unique_ptr<KalmanTracklet>> tracklets_;
};

static_assert(TrackerType<SortTracker>);

} // namespace improc::ml
