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
    const std::size_t n_existing = tracks_.size();
    std::vector<bool> det_used(dets.size(), false);
    std::vector<bool> trk_used(n_existing, false);

    // Greedy: repeatedly pick best unmatched pair above min_iou_
    while (true) {
        float best = min_iou_;
        int bi = -1, bj = -1;
        for (std::size_t i = 0; i < dets.size(); ++i) {
            if (det_used[i]) continue;
            for (std::size_t j = 0; j < n_existing; ++j) {
                if (trk_used[j]) continue;
                BBox det_box{dets[i].box, dets[i].class_id, dets[i].label};
                float s = bbox_iou(det_box, tracks_[j].bbox);
                if (s > best) { best = s; bi = static_cast<int>(i); bj = static_cast<int>(j); }
            }
        }
        if (bi < 0) break;

        auto& t = tracks_[bj];
        t.track.bbox       = BBox{dets[bi].box, dets[bi].class_id, dets[bi].label};
        t.track.confidence = dets[bi].confidence;
        t.track.age        = 0;
        t.bbox             = t.track.bbox;
        det_used[bi] = trk_used[bj] = true;
    }

    // Unmatched detections → new confirmed tracks
    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (det_used[i]) continue;
        InternalTrack it;
        it.track.id           = next_id_++;
        it.track.bbox         = BBox{dets[i].box, dets[i].class_id, dets[i].label};
        it.track.confidence   = dets[i].confidence;
        it.track.age          = 0;
        it.track.is_confirmed = true;
        it.bbox               = it.track.bbox;
        tracks_.push_back(it);
    }

    // Age unmatched existing tracks (not newly added ones)
    for (std::size_t j = 0; j < n_existing; ++j)
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
