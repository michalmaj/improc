// src/ml/sort_tracker.cpp
#include "improc/ml/tracking/sort_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/video/tracking.hpp>

namespace improc::ml {

// ── bbox helpers ──────────────────────────────────────────────────────────────

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

cv::Mat to_meas(const BBox& b) {
    float cx = b.box.x + b.box.width  / 2.0f;
    float cy = b.box.y + b.box.height / 2.0f;
    float s  = b.box.width * b.box.height;
    float r  = b.box.height > 0.0f ? b.box.width / b.box.height : 1.0f;
    return (cv::Mat_<float>(4,1) << cx, cy, s, r);
}

BBox from_state(const cv::Mat& state, int class_id = 0,
                const std::string& label = "") {
    float cx = state.at<float>(0);
    float cy = state.at<float>(1);
    float s  = std::max(0.0f, state.at<float>(2));
    float r  = std::max(1e-4f, state.at<float>(3));
    float w  = std::sqrt(s * r);
    float h  = w > 0.0f ? s / w : 0.0f;
    return BBox{cv::Rect2f(cx - w / 2.0f, cy - h / 2.0f, w, h), class_id, label};
}

// ── O(n³) Hungarian algorithm ──────────────────────────────────────────────────
// cost[i][j] = cost of assigning row i to col j (minimize).
// Returns assignment[i] = j (row i → col j), -1 if unassigned.

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
    std::vector<int>   p(sz+1, 0), way(sz+1, 0);

    for (int i = 1; i <= sz; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(sz+1, kInf);
        std::vector<bool>  used(sz+1, false);
        do {
            used[j0] = true;
            int   i0 = p[j0], j1 = -1;
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

    std::vector<int> assignment(n, -1);
    for (int j = 1; j <= sz && j <= m; ++j)
        if (p[j] > 0 && p[j] <= n) assignment[p[j]-1] = j-1;
    return assignment;
}

} // anonymous namespace

// ── Kalman tracklet ─────────────────────────────────────────────────────────────

struct SortTracker::KalmanTracklet {
    cv::KalmanFilter kf{7, 4, 0, CV_32F};
    int   id;
    int   age  = 0;
    int   hits = 1;  // creation counts as first hit
    bool  confirmed = false;
    BBox  predicted_bbox;
    int   class_id = 0;
    std::string label;

    KalmanTracklet(int track_id, const Detection& det)
        : id(track_id), class_id(det.class_id), label(det.label) {
        kf.transitionMatrix = (cv::Mat_<float>(7,7) <<
            1,0,0,0,1,0,0,
            0,1,0,0,0,1,0,
            0,0,1,0,0,0,1,
            0,0,0,1,0,0,0,
            0,0,0,0,1,0,0,
            0,0,0,0,0,1,0,
            0,0,0,0,0,0,1);
        kf.measurementMatrix = cv::Mat::eye(4, 7, CV_32F);
        cv::setIdentity(kf.processNoiseCov, cv::Scalar(1e-2));
        kf.processNoiseCov.at<float>(4,4) = 1e-2f;
        kf.processNoiseCov.at<float>(5,5) = 1e-2f;
        kf.processNoiseCov.at<float>(6,6) = 1e-4f;
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1.0f));
        kf.measurementNoiseCov.at<float>(2,2) = 10.0f;
        cv::setIdentity(kf.errorCovPost, cv::Scalar(10.0f));
        kf.errorCovPost.at<float>(4,4) = 1e4f;
        kf.errorCovPost.at<float>(5,5) = 1e4f;
        kf.errorCovPost.at<float>(6,6) = 1e4f;
        BBox b{det.box, det.class_id, det.label};
        auto m = to_meas(b);
        for (int k = 0; k < 4; ++k) kf.statePost.at<float>(k) = m.at<float>(k);
        for (int k = 4; k < 7; ++k) kf.statePost.at<float>(k) = 0.0f;
        predicted_bbox = b;
    }

    void predict() {
        auto pred = kf.predict();
        predicted_bbox = from_state(pred, class_id, label);
        age++;
    }

    void correct(const Detection& det) {
        BBox b{det.box, det.class_id, det.label};
        kf.correct(to_meas(b));
        predicted_bbox = b;
        class_id = det.class_id;
        label    = det.label;
        hits++;
        age = 0;
    }

    Track to_track() const {
        return Track{id, predicted_bbox, 0.0f, age, confirmed};
    }
};

// ── SortTracker ctor/dtor (defined here so KalmanTracklet is complete) ─────────

SortTracker::SortTracker()  = default;
SortTracker::~SortTracker() = default;

// ── SortTracker::update ────────────────────────────────────────────────────────

std::vector<Track> SortTracker::update(const std::vector<Detection>& dets) {
    // 1. Predict all tracklets
    for (auto& trk : tracklets_) trk->predict();

    // 2. Build cost matrix (1 - IoU) [n_dets × n_tracklets]
    int nd = static_cast<int>(dets.size());
    int nt = static_cast<int>(tracklets_.size());
    std::vector<std::vector<float>> cost(nd, std::vector<float>(nt, 1.0f));
    for (int i = 0; i < nd; ++i)
        for (int j = 0; j < nt; ++j) {
            BBox det_box{dets[i].box, dets[i].class_id, dets[i].label};
            float s = bbox_iou(det_box, tracklets_[j]->predicted_bbox);
            cost[i][j] = 1.0f - s;
        }

    // 3. Hungarian assignment
    auto det_to_trk = (nd > 0 && nt > 0) ? hungarian(cost) : std::vector<int>(nd, -1);

    for (int i = 0; i < nd; ++i) {
        int j = (i < static_cast<int>(det_to_trk.size())) ? det_to_trk[i] : -1;
        if (j >= 0 && j < nt) {
            BBox det_box{dets[i].box, dets[i].class_id, dets[i].label};
            float s = bbox_iou(det_box, tracklets_[j]->predicted_bbox);
            if (s >= iou_thr_) {
                tracklets_[j]->correct(dets[i]);
                continue;
            }
        }
        // Unmatched detection → new tracklet
        tracklets_.push_back(std::make_unique<KalmanTracklet>(next_id_++, dets[i]));
    }

    // 4. Remove dead tracklets
    std::erase_if(tracklets_, [this](const std::unique_ptr<KalmanTracklet>& t) {
        return t->age > max_age_;
    });

    // 5. Collect output
    for (auto& t : tracklets_)
        if (t->hits >= min_hits_) t->confirmed = true;

    std::vector<Track> result;
    for (const auto& t : tracklets_)
        if (t->confirmed) result.push_back(t->to_track());
    return result;
}

void SortTracker::reset() {
    tracklets_.clear();
    next_id_ = 0;
}

} // namespace improc::ml
