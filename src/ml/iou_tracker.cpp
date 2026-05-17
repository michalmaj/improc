// src/ml/iou_tracker.cpp
#include "improc/ml/tracking/iou_tracker.hpp"
#include <algorithm>

namespace improc::ml {

namespace {
float bbox_iou(const BBox& a, const BBox& b) {
    float ix1 = std::max(a.box.x, b.box.x);
    float iy1 = std::max(a.box.y, b.box.y);
    float ix2 = std::min(a.box.x + a.box.width,  b.box.x + b.box.width);
    float iy2 = std::min(a.box.y + a.box.height, b.box.y + b.box.height);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float area_a = a.box.width * a.box.height;
    float area_b = b.box.width * b.box.height;
    float uni = area_a + area_b - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}
} // namespace

std::vector<Track> IouTracker::update(const std::vector<Detection>& dets) {
    const std::size_t D = dets.size();
    const std::size_t T = tracks_.size();

    // Pre-convert detections to BBox once — avoids repeated string allocations
    // inside the matching loop.
    std::vector<BBox> det_boxes;
    det_boxes.reserve(D);
    for (const auto& d : dets)
        det_boxes.push_back({d.box, d.class_id, d.label});

    std::vector<bool> det_used(D, false);
    std::vector<bool> trk_used(T, false);

    if (D > 0 && T > 0) {
        // Build IoU matrix once (D×T) — previous code recomputed it min(D,T) times.
        std::vector<float> iou_mat(D * T);
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < T; ++j)
                iou_mat[i * T + j] = bbox_iou(det_boxes[i], tracks_[j].bbox);

        // Collect above-threshold pairs, sort by IoU descending, then greedily assign.
        struct Pair { float iou; std::size_t di, ti; };
        std::vector<Pair> pairs;
        pairs.reserve(D * T);
        for (std::size_t i = 0; i < D; ++i)
            for (std::size_t j = 0; j < T; ++j)
                if (iou_mat[i * T + j] > min_iou_)
                    pairs.push_back({iou_mat[i * T + j], i, j});
        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair& a, const Pair& b) { return a.iou > b.iou; });

        for (const auto& p : pairs) {
            if (det_used[p.di] || trk_used[p.ti]) continue;
            auto& t = tracks_[p.ti];
            t.track.bbox       = det_boxes[p.di];
            t.track.confidence = dets[p.di].confidence;
            t.track.age        = 0;
            t.bbox             = t.track.bbox;
            det_used[p.di] = trk_used[p.ti] = true;
        }
    }

    // Unmatched detections → new confirmed tracks
    for (std::size_t i = 0; i < D; ++i) {
        if (det_used[i]) continue;
        InternalTrack it;
        it.track.id           = next_id_++;
        it.track.bbox         = det_boxes[i];
        it.track.confidence   = dets[i].confidence;
        it.track.age          = 0;
        it.track.is_confirmed = true;
        it.bbox               = it.track.bbox;
        tracks_.push_back(it);
    }

    // Age unmatched existing tracks (not newly added ones)
    for (std::size_t j = 0; j < T; ++j)
        if (!trk_used[j]) tracks_[j].track.age++;

    // Remove dead tracks
    std::erase_if(tracks_, [this](const InternalTrack& t) {
        return t.track.age > max_age_;
    });

    std::vector<Track> result;
    result.reserve(tracks_.size());
    for (const auto& it : tracks_) result.push_back(it.track);
    return result;
}

void IouTracker::reset() {
    tracks_.clear();
    next_id_ = 0;
}

} // namespace improc::ml
