// src/ml/byte_tracker.cpp
#include "improc/ml/tracking/byte_tracker.hpp"
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

BBox from_state(const cv::Mat& st, int class_id, const std::string& label) {
    float cx = st.at<float>(0), cy = st.at<float>(1);
    float s  = std::max(0.0f, st.at<float>(2));
    float r  = std::max(1e-4f, st.at<float>(3));
    float w  = std::sqrt(s * r), h = w > 0.0f ? s / w : 0.0f;
    return BBox{cv::Rect2f(cx - w/2.0f, cy - h/2.0f, w, h), class_id, label};
}

// O(n³) Hungarian algorithm — cost[i][j] = cost of assigning row i to col j (minimize).
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

} // anonymous namespace

// ── Kalman tracklet ─────────────────────────────────────────────────────────────

struct ByteTracker::KalmanTracklet {
    cv::KalmanFilter kf{7, 4, 0, CV_32F};
    int id;
    int age      = 0;
    int hits      = 1;  // creation counts as first hit
    bool confirmed = false;
    BBox predicted_bbox;
    int class_id  = 0;
    std::string label;

    KalmanTracklet(int tid, const Detection& det)
        : id(tid), class_id(det.class_id), label(det.label) {
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

    // Low-conf match: use detection bbox directly, no Kalman correction
    void correct_bbox(const BBox& b) {
        predicted_bbox = b;
        hits++;
        age = 0;
    }

    Track to_track() const {
        return Track{id, predicted_bbox, 0.0f, age, confirmed};
    }
};

// ── ByteTracker ctor/dtor (defined here so KalmanTracklet is complete) ─────────

ByteTracker::ByteTracker()  = default;
ByteTracker::~ByteTracker() = default;

// ── ByteTracker::update ────────────────────────────────────────────────────────

std::vector<Track> ByteTracker::update(const std::vector<Detection>& dets) {
    // 1. Predict all existing tracklets
    for (auto& t : tracklets_) t->predict();

    // Capture count before any additions
    const int nt = static_cast<int>(tracklets_.size());

    // 2. Split detections into high-conf and low-conf buckets
    std::vector<Detection> high_dets, low_dets;
    for (const auto& d : dets) {
        if      (d.confidence >= high_thr_) high_dets.push_back(d);
        else if (d.confidence >= low_thr_)  low_dets.push_back(d);
        // below low_thr_ → discarded
    }

    std::vector<bool> trk_matched(nt, false);

    // 3. Stage 1: Hungarian on high-conf dets vs all existing tracklets
    if (!high_dets.empty() && nt > 0) {
        int nd = static_cast<int>(high_dets.size());
        std::vector<std::vector<float>> cost(nd, std::vector<float>(nt, 1.0f));
        for (int i = 0; i < nd; ++i)
            for (int j = 0; j < nt; ++j) {
                BBox b{high_dets[i].box, high_dets[i].class_id, high_dets[i].label};
                cost[i][j] = 1.0f - bbox_iou(b, tracklets_[j]->predicted_bbox);
            }
        auto asgn = hungarian(cost);
        std::vector<bool> det_matched(nd, false);
        for (int i = 0; i < nd; ++i) {
            int j = (i < static_cast<int>(asgn.size())) ? asgn[i] : -1;
            if (j >= 0 && j < nt) {
                BBox b{high_dets[i].box, high_dets[i].class_id, high_dets[i].label};
                if (bbox_iou(b, tracklets_[j]->predicted_bbox) >= 0.3f) {
                    tracklets_[j]->correct(high_dets[i]);
                    trk_matched[j] = det_matched[i] = true;
                }
            }
        }
        // Unmatched high-conf dets → new tracklets
        for (int i = 0; i < nd; ++i)
            if (!det_matched[i])
                tracklets_.push_back(std::make_unique<KalmanTracklet>(next_id_++, high_dets[i]));
    } else {
        // No existing tracklets: all high-conf dets become new tracklets
        for (const auto& d : high_dets)
            tracklets_.push_back(std::make_unique<KalmanTracklet>(next_id_++, d));
    }

    // 4. Stage 2: Greedy IoU — unmatched existing tracklets vs low-conf dets
    // Only iterate over the first nt tracklets (new ones from stage 1 are already handled).
    // This also avoids out-of-bounds access on trk_matched which has size nt.
    for (int idx = 0; idx < nt; ++idx) {
        if (trk_matched[idx]) continue;
        float best   = 0.3f;  // minimum IoU threshold for low-conf match
        int   best_d = -1;
        for (std::size_t d = 0; d < low_dets.size(); ++d) {
            BBox b{low_dets[d].box, low_dets[d].class_id, low_dets[d].label};
            float s = bbox_iou(b, tracklets_[idx]->predicted_bbox);
            if (s > best) { best = s; best_d = static_cast<int>(d); }
        }
        if (best_d >= 0) {
            tracklets_[idx]->correct_bbox(
                BBox{low_dets[best_d].box, low_dets[best_d].class_id, low_dets[best_d].label});
            trk_matched[idx] = true;
        }
    }

    // 5. Remove dead tracklets (age > max_age_ means no match for too long)
    std::erase_if(tracklets_, [this](const std::unique_ptr<KalmanTracklet>& t) {
        return t->age > max_age_;
    });

    // 6. Confirm tracklets that have enough hits
    for (auto& t : tracklets_)
        if (t->hits >= min_hits_) t->confirmed = true;

    // 7. Collect and return confirmed tracks
    std::vector<Track> result;
    for (const auto& t : tracklets_)
        if (t->confirmed) result.push_back(t->to_track());
    return result;
}

void ByteTracker::reset() {
    tracklets_.clear();
    next_id_ = 0;
}

} // namespace improc::ml
