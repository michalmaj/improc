// tests/ml/test_iou_tracker.cpp
#include <gtest/gtest.h>
#include "improc/ml/tracking/track.hpp"
#include "improc/ml/tracking/iou_tracker.hpp"

using namespace improc::ml;

static Detection make_det(float x, float y, float w, float h,
                           float conf = 1.0f, std::string lbl = "car") {
    return Detection{cv::Rect2f(x, y, w, h), 0, conf, std::move(lbl)};
}

TEST(IouTrackerTest, TrackTest_DefaultConstruct) {
    Track t;
    EXPECT_EQ(t.id, -1);
}

TEST(IouTrackerTest, PerfectOverlapKeepsId) {
    IouTracker tracker;
    auto dets1 = std::vector<Detection>{make_det(0, 0, 10, 10)};
    auto r1 = tracker.update(dets1);
    ASSERT_EQ(r1.size(), 1u);
    int first_id = r1[0].id;

    auto dets2 = std::vector<Detection>{make_det(0, 0, 10, 10)};
    auto r2 = tracker.update(dets2);
    ASSERT_EQ(r2.size(), 1u);
    EXPECT_EQ(r2[0].id, first_id);
}

TEST(IouTrackerTest, NoOverlapCreatesNewTracks) {
    IouTracker tracker;
    tracker.max_age(0);
    auto r1 = tracker.update({make_det(0, 0, 10, 10)});
    ASSERT_EQ(r1.size(), 1u);
    int id1 = r1[0].id;

    // Move far away so IoU = 0
    auto r2 = tracker.update({make_det(100, 100, 10, 10)});
    ASSERT_EQ(r2.size(), 1u);
    EXPECT_NE(r2[0].id, id1);
}

TEST(IouTrackerTest, TrackDiesAfterMaxAge) {
    IouTracker tracker;
    tracker.max_age(1);
    tracker.update({make_det(0, 0, 10, 10)});  // frame 1: track created
    tracker.update({});                          // frame 2: age=1, alive
    auto r3 = tracker.update({});               // frame 3: age=2 > max_age=1 → removed
    EXPECT_TRUE(r3.empty());
}

TEST(IouTrackerTest, MultipleObjectsIndependent) {
    IouTracker tracker;
    auto r = tracker.update({
        make_det(0,   0, 10, 10),
        make_det(100, 0, 10, 10),
        make_det(0, 100, 10, 10)
    });
    EXPECT_EQ(r.size(), 3u);
    EXPECT_NE(r[0].id, r[1].id);
    EXPECT_NE(r[0].id, r[2].id);
    EXPECT_NE(r[1].id, r[2].id);
}

TEST(IouTrackerTest, ResetClearsState) {
    IouTracker tracker;
    tracker.update({make_det(0, 0, 10, 10)});
    tracker.reset();
    auto r = tracker.update({make_det(0, 0, 10, 10)});
    ASSERT_EQ(r.size(), 1u);
    EXPECT_EQ(r[0].id, 0);  // ID counter reset to 0
}
