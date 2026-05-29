// include/improc/ml/tracking/iou_tracker.hpp
#pragma once
#include "improc/ml/tracking/track.hpp"
#include "improc/exceptions.hpp"

namespace improc::ml {

/**
 * @brief Greedy IoU-based tracker with no motion model.
 *
 * Associates each incoming detection to the nearest existing track by
 * intersection-over-union (IoU), using a greedy nearest-first strategy.
 * No Kalman filter — bounding boxes are not predicted between frames.
 * Tracks disappear after `max_age` consecutive frames without a match.
 *
 * Best for: high-framerate, low-occlusion scenarios where detections are
 * reliable every frame. Falls back to poor performance under occlusion or
 * missed detections.
 *
 * @code
 * IouTracker tracker;
 * tracker.min_iou(0.4f).max_age(2);
 * std::vector<Track> tracks = tracker.update(detections);
 * @endcode
 */
struct IouTracker {
    /// @brief Sets the minimum IoU overlap required to associate a detection to a track (default: 0.3).
    /// @throws improc::ParameterError if `t` is outside [0, 1].
    IouTracker& min_iou(float t) {
        if (t < 0.f || t > 1.f)
            throw improc::ParameterError("min_iou", "must be in [0, 1]", "IouTracker");
        min_iou_ = t; return *this;
    }
    /// @brief Sets the number of consecutive unmatched frames before a track is deleted (default: 1).
    /// @throws improc::ParameterError if `frames` < 0.
    IouTracker& max_age(int frames) {
        if (frames < 0)
            throw improc::ParameterError("max_age", "must be >= 0", "IouTracker");
        max_age_ = frames; return *this;
    }

    /// @brief Processes one frame of detections and returns currently active tracks.
    [[nodiscard]] std::vector<Track> update(const std::vector<Detection>& dets);
    /// @brief Clears all tracks and resets the ID counter.
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
