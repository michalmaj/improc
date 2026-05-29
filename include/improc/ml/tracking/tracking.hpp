/**
 * @brief Umbrella include for all `improc::ml` tracking types and algorithms.
 *
 * Including this header pulls in:
 * - `Track`, `TrackGT`, `TrackerType` concept (`track.hpp`)
 * - `IouTracker` — greedy IoU assignment, no Kalman filter (`iou_tracker.hpp`)
 * - `SortTracker` — Kalman + Hungarian algorithm (SORT) (`sort_tracker.hpp`)
 * - `ByteTracker` — two-stage Kalman association (ByteTrack) (`byte_tracker.hpp`)
 * - `TrackingEval`, `TrackingMetrics` — MOTA/MOTP/IDF1 accumulator (`tracking_eval.hpp`)
 *
 * @code
 * #include "improc/ml/tracking/tracking.hpp"
 * using namespace improc::ml;
 *
 * ByteTracker tracker;
 * tracker.max_age(5).min_hits(2).high_conf_threshold(0.6f);
 *
 * for (auto& frame : sequence) {
 *     std::vector<Detection> dets = detector(frame);
 *     std::vector<Track> tracks  = tracker.update(dets);
 * }
 * @endcode
 */
#pragma once

#include "improc/ml/tracking/track.hpp"
#include "improc/ml/tracking/iou_tracker.hpp"
#include "improc/ml/tracking/sort_tracker.hpp"
#include "improc/ml/tracking/byte_tracker.hpp"
#include "improc/ml/tracking/tracking_eval.hpp"
