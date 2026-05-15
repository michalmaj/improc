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
    for (const auto& t : r1) EXPECT_FALSE(t.is_confirmed);
    for (const auto& t : r2) EXPECT_FALSE(t.is_confirmed);
    auto r3 = tracker.update({make_det(0, 0, 10, 10)});
    bool any_confirmed = false;
    for (const auto& t : r3) if (t.is_confirmed) any_confirmed = true;
    EXPECT_TRUE(any_confirmed);
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
    tracker.update({make_det(0, 0, 10, 10)});
    auto r1 = tracker.update({make_det(0, 0, 10, 10)});
    int saved_id = r1.empty() ? -1 : r1[0].id;

    tracker.update({});  // occluded
    tracker.update({});  // still occluded (age=2 <= max_age=3)
    auto r4 = tracker.update({make_det(0, 0, 10, 10)});
    ASSERT_FALSE(r4.empty());
    EXPECT_EQ(r4[0].id, saved_id);
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
    tracker.update({make_det(0,  0, 10, 10)});
    tracker.update({make_det(5,  0, 10, 10)});
    tracker.update({make_det(10, 0, 10, 10)});

    auto r = tracker.update({});
    ASSERT_FALSE(r.empty());
    float pred_cx = r[0].bbox.box.x + r[0].bbox.box.width / 2.0f;
    EXPECT_GT(pred_cx, 10.0f);
}

TEST(SortTrackerTest, ResetClearsState) {
    SortTracker tracker;
    tracker.min_hits(1);
    tracker.update({make_det(0, 0, 10, 10)});
    tracker.update({make_det(0, 0, 10, 10)});
    tracker.reset();
    auto r = tracker.update({make_det(0, 0, 10, 10)});
    for (const auto& t : r) EXPECT_EQ(t.id, 0);
}
