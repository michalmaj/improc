// tests/ml/test_sort_tracker.cpp
#include <gtest/gtest.h>
#include "improc/ml/tracking/sort_tracker.hpp"

using namespace improc::ml;

static Detection make_det(float x, float y, float w, float h, float conf = 1.0f) {
    return Detection{cv::Rect2f(x, y, w, h), 0, conf, "car"};
}

TEST(SortTrackerTest, ConfirmationThreshold) {
    SortTracker tracker;
    tracker.min_hits(3).max_age(5);
    auto r1 = tracker.update({make_det(0, 0, 10, 10)});
    auto r2 = tracker.update({make_det(0, 0, 10, 10)});
    EXPECT_EQ(r1.size(), 0u);  // unconfirmed on frame 1
    EXPECT_EQ(r2.size(), 0u);  // unconfirmed on frame 2
    auto r3 = tracker.update({make_det(0, 0, 10, 10)});
    ASSERT_EQ(r3.size(), 1u);
    EXPECT_TRUE(r3[0].is_confirmed);
}

TEST(SortTrackerTest, IdStabilityOverFrames) {
    SortTracker tracker;
    tracker.min_hits(1).max_age(5);
    int first_id = -1;
    for (int frame = 0; frame < 5; ++frame) {
        float x = static_cast<float>(frame) * 2.0f;
        auto r = tracker.update({make_det(x, 0, 10, 10)});
        if (r.empty()) continue;
        if (first_id < 0) first_id = r[0].id;
        else EXPECT_EQ(r[0].id, first_id) << "ID changed at frame " << frame;
    }
}

TEST(SortTrackerTest, OcclusionRecovery) {
    SortTracker tracker;
    tracker.min_hits(1).max_age(3);

    // Build a confirmed track
    (void)tracker.update({make_det(0, 0, 10, 10)});
    auto r = tracker.update({make_det(0, 0, 10, 10)});
    ASSERT_FALSE(r.empty());
    int saved_id = r[0].id;

    // Blank frames 1..max_age: track must survive every one
    for (int i = 1; i <= 3; ++i) {
        auto blank = tracker.update({});
        EXPECT_FALSE(blank.empty()) << "track should survive blank frame " << i;
    }

    // One more blank frame (max_age+1): track must be removed
    auto dead = tracker.update({});
    EXPECT_TRUE(dead.empty()) << "track should be removed after max_age+1 blank frames";
}

TEST(SortTrackerTest, MultipleObjectsNoIdSwitch) {
    SortTracker tracker;
    tracker.min_hits(1).max_age(3).iou_threshold(0.1f);

    std::vector<int> ids;
    for (int frame = 0; frame < 5; ++frame) {
        auto r = tracker.update({
            make_det(0.0f + frame * 1.0f,   0, 10, 10),
            make_det(100.0f + frame * 1.0f, 0, 10, 10),
            make_det(200.0f + frame * 1.0f, 0, 10, 10)
        });
        if (frame == 0) for (const auto& t : r) ids.push_back(t.id);
    }

    auto final_r = tracker.update({
        make_det(5, 0, 10, 10), make_det(105, 0, 10, 10), make_det(205, 0, 10, 10)
    });
    std::sort(ids.begin(), ids.end());
    std::vector<int> final_ids;
    for (const auto& t : final_r) if (t.is_confirmed) final_ids.push_back(t.id);
    std::sort(final_ids.begin(), final_ids.end());
    EXPECT_EQ(ids, final_ids);
}

TEST(SortTrackerTest, KalmanPredictsMissingDetection) {
    SortTracker tracker;
    tracker.min_hits(1).max_age(5);
    (void)tracker.update({make_det(0,  0, 10, 10)});
    (void)tracker.update({make_det(5,  0, 10, 10)});
    (void)tracker.update({make_det(10, 0, 10, 10)});

    auto r = tracker.update({});
    ASSERT_FALSE(r.empty());
    float pred_cx = r[0].bbox.box.x + r[0].bbox.box.width / 2.0f;
    EXPECT_GT(pred_cx, 10.0f);
}

TEST(SortTrackerTest, ResetClearsState) {
    SortTracker tracker;
    tracker.min_hits(1);
    (void)tracker.update({make_det(0, 0, 10, 10)});
    (void)tracker.update({make_det(0, 0, 10, 10)});
    tracker.reset();
    auto r = tracker.update({make_det(0, 0, 10, 10)});
    for (const auto& t : r) EXPECT_EQ(t.id, 0);
}
