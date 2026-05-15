// src/ml/tracking_eval.cpp
#include "improc/ml/tracking/tracking_eval.hpp"
#include <algorithm>
#include <cmath>
#include <set>

namespace improc::ml {

namespace {

float bbox_iou(const BBox& a, const BBox& b) {
    float ix1 = std::max(a.box.x, b.box.x);
    float iy1 = std::max(a.box.y, b.box.y);
    float ix2 = std::min(a.box.x + a.box.width,  b.box.x + b.box.width);
    float iy2 = std::min(a.box.y + a.box.height, b.box.y + b.box.height);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float uni = a.box.width * a.box.height + b.box.width * b.box.height - inter;
    return uni <= 0.0f ? 0.0f : inter / uni;
}

// O(n^3) Hungarian — same algorithm as sort_tracker.cpp/byte_tracker.cpp.
// Anonymous namespace: no linkage conflict with other translation units.
std::vector<int> hungarian(const std::vector<std::vector<float>>& cost) {
    int n = static_cast<int>(cost.size());
    if (n == 0) return {};
    int m = static_cast<int>(cost[0].size());
    if (m == 0) return std::vector<int>(n, -1);
    int sz = std::max(n, m);
    const float kInf = 1e9f;
    std::vector<std::vector<float>> C(sz, std::vector<float>(sz, kInf));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            C[i][j] = cost[i][j];
    std::vector<float> u(sz+1, 0.f), v(sz+1, 0.f);
    std::vector<int>   p(sz+1, 0),   way(sz+1, 0);
    for (int i = 1; i <= sz; ++i) {
        p[0] = i; int j0 = 0;
        std::vector<float> minv(sz+1, kInf);
        std::vector<bool>  used(sz+1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = -1;
            float delta = kInf;
            for (int j = 1; j <= sz; ++j) {
                if (!used[j]) {
                    float cur = C[i0-1][j-1] - u[i0] - v[j];
                    if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            for (int j = 0; j <= sz; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    std::vector<int> asgn(n, -1);
    for (int j = 1; j <= sz && j <= m; ++j)
        if (p[j] > 0 && p[j] <= n) asgn[p[j]-1] = j-1;
    return asgn;
}

} // namespace

void TrackingEval::update(const std::vector<Track>&   tracks,
                          const std::vector<TrackGT>& gts) {
    GT_total_ += static_cast<int>(gts.size());
    for (const auto& gt : gts) gt_total_[gt.id]++;

    // Greedy match: for each track find best unmatched GT by IoU
    std::vector<bool> gt_matched(gts.size(), false);

    for (const auto& trk : tracks) {
        float best   = iou_thr_;
        int   best_j = -1;
        for (std::size_t j = 0; j < gts.size(); ++j) {
            if (gt_matched[j]) continue;
            float s = bbox_iou(trk.bbox, gts[j].bbox);
            if (s > best) { best = s; best_j = static_cast<int>(j); }
        }
        if (best_j >= 0) {
            gt_matched[best_j] = true;
            iou_sum_ += best;
            ++match_count_;
            int gt_id = gts[best_j].id;

            // ID switch: same GT was previously matched by a different track
            auto it = last_track_for_gt_.find(gt_id);
            if (it != last_track_for_gt_.end() && it->second != trk.id) ++IDSW_;
            last_track_for_gt_[gt_id] = trk.id;

            pair_frames_[{trk.id, gt_id}]++;
            track_total_[trk.id]++;
        } else {
            ++FP_;
            track_total_[trk.id]++;
        }
    }
    for (std::size_t j = 0; j < gts.size(); ++j)
        if (!gt_matched[j]) ++FN_;
}

TrackingMetrics TrackingEval::compute() const {
    TrackingMetrics m;
    m.FP   = FP_;
    m.FN   = FN_;
    m.IDSW = IDSW_;
    m.MOTA = GT_total_ > 0
        ? 1.0f - static_cast<float>(FN_ + FP_ + IDSW_) / GT_total_
        : 0.0f;
    m.MOTP = match_count_ > 0
        ? iou_sum_ / static_cast<float>(match_count_)
        : 0.0f;

    // IDF1 via global track<->GT bipartite matching on co-occurrence counts
    std::set<int> all_tids, all_gids;
    for (const auto& [tid, _] : track_total_) all_tids.insert(tid);
    for (const auto& [gid, _] : gt_total_)   all_gids.insert(gid);

    if (all_tids.empty() || all_gids.empty()) { m.IDF1 = 0.0f; return m; }

    std::vector<int> tids(all_tids.begin(), all_tids.end());
    std::vector<int> gids(all_gids.begin(), all_gids.end());
    int nt = static_cast<int>(tids.size());
    int ng = static_cast<int>(gids.size());

    // Cost = -co_occurrence (Hungarian minimizes, so negate to maximize)
    std::vector<std::vector<float>> cost(nt, std::vector<float>(ng, 0.0f));
    for (int i = 0; i < nt; ++i)
        for (int j = 0; j < ng; ++j) {
            auto it = pair_frames_.find({tids[i], gids[j]});
            cost[i][j] = -(it != pair_frames_.end() ? static_cast<float>(it->second) : 0.0f);
        }

    auto asgn = hungarian(cost);

    int64_t IDTP = 0, IDFP = 0, IDFN = 0;
    for (int i = 0; i < nt; ++i) {
        int j = (i < static_cast<int>(asgn.size())) ? asgn[i] : -1;
        if (j >= 0 && j < ng) {
            auto it = pair_frames_.find({tids[i], gids[j]});
            int64_t tp = it != pair_frames_.end() ? it->second : 0;
            IDTP += tp;
            IDFP += track_total_.at(tids[i]) - tp;
            IDFN += gt_total_.at(gids[j])    - tp;
        } else {
            IDFP += track_total_.at(tids[i]);
        }
    }
    // Unmatched GTs contribute to IDFN
    for (int j = 0; j < ng; ++j) {
        bool matched_by_any = false;
        for (int i = 0; i < nt; ++i)
            if (i < static_cast<int>(asgn.size()) && asgn[i] == j) {
                matched_by_any = true; break;
            }
        if (!matched_by_any) IDFN += gt_total_.at(gids[j]);
    }

    int64_t denom = 2*IDTP + IDFP + IDFN;
    m.IDF1 = denom > 0 ? static_cast<float>(2*IDTP) / static_cast<float>(denom) : 0.0f;
    return m;
}

void TrackingEval::reset() {
    FP_ = FN_ = IDSW_ = GT_total_ = match_count_ = 0;
    iou_sum_ = 0.0f;
    last_track_for_gt_.clear();
    track_total_.clear();
    gt_total_.clear();
    pair_frames_.clear();
}

} // namespace improc::ml
