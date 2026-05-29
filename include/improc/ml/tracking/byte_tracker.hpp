// include/improc/ml/tracking/byte_tracker.hpp
#pragma once
#include "improc/ml/tracking/track.hpp"
#include "improc/exceptions.hpp"
#include <memory>

namespace improc::ml {

/**
 * @brief ByteTrack: two-stage Kalman association using high- and low-confidence detections.
 *
 * Stage 1 matches tracks against high-confidence detections (≥ `high_conf_threshold`)
 * via the Hungarian algorithm on IoU costs. Stage 2 gives remaining unmatched tracks a
 * second chance against low-confidence detections (≥ `low_conf_threshold`), recovering
 * tracks temporarily occluded by a low-scoring detector response. Kalman filtering
 * predicts each track's next position between frames. Tracks are promoted to "confirmed"
 * after `min_hits` consecutive matches.
 *
 * Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
 * Detection Box", ECCV 2022.
 *
 * @code
 * ByteTracker tracker;
 * tracker.max_age(5).min_hits(2).high_conf_threshold(0.6f).low_conf_threshold(0.1f);
 * std::vector<Track> tracks = tracker.update(detections);
 * @endcode
 */
struct ByteTracker {
    /// @brief Sets the number of consecutive unmatched frames before a track is deleted (default: 3).
    /// @throws improc::ParameterError if `frames` < 0.
    ByteTracker& max_age(int frames) {
        if (frames < 0)
            throw improc::ParameterError("max_age", "must be >= 0", "ByteTracker");
        max_age_ = frames; return *this;
    }
    /// @brief Sets the minimum consecutive matches required before a track is confirmed (default: 3).
    /// @throws improc::ParameterError if `hits` < 1.
    ByteTracker& min_hits(int hits) {
        if (hits < 1)
            throw improc::ParameterError("min_hits", "must be >= 1", "ByteTracker");
        min_hits_ = hits; return *this;
    }
    /// @brief Sets the confidence threshold for stage-1 (primary) association (default: 0.6).
    /// @throws improc::ParameterError if `t` is outside [0, 1].
    ByteTracker& high_conf_threshold(float t) {
        if (t < 0.f || t > 1.f)
            throw improc::ParameterError("high_conf_threshold", "must be in [0, 1]", "ByteTracker");
        high_thr_ = t; return *this;
    }
    /// @brief Sets the confidence threshold for stage-2 (recovery) association (default: 0.1).
    /// @throws improc::ParameterError if `t` is outside [0, 1].
    ByteTracker& low_conf_threshold(float t) {
        if (t < 0.f || t > 1.f)
            throw improc::ParameterError("low_conf_threshold", "must be in [0, 1]", "ByteTracker");
        low_thr_ = t; return *this;
    }

    /// @brief Processes one frame of detections and returns currently active tracks.
    [[nodiscard]] std::vector<Track> update(const std::vector<Detection>& dets);
    /// @brief Clears all tracks and resets the ID counter.
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
