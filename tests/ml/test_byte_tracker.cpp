// tests/ml/test_byte_tracker.cpp
#include <gtest/gtest.h>
#include "improc/ml/tracking/byte_tracker.hpp"
#include "improc/ml/tracking/sort_tracker.hpp"
#include "improc/exceptions.hpp"

using namespace improc::ml;

static Detection make_det(float x, float y, float w, float h, float conf) {
    return Detection{cv::Rect2f(x, y, w, h), 0, conf, "car"};
}

TEST(ByteTrackerTest, LowConfRecovery) {
    ByteTracker tracker;
    tracker.min_hits(1).max_age(0)
           .high_conf_threshold(0.6f).low_conf_threshold(0.1f);

    tracker.update({make_det(0, 0, 10, 10, 0.9f)});
    auto r1 = tracker.update({make_det(2, 0, 10, 10, 0.9f)});
    ASSERT_FALSE(r1.empty());
    int saved_id = r1[0].id;

    // Frame 3: only a low-conf detection — Stage 1 has no high dets, Stage 2 must match
    auto r2 = tracker.update({make_det(4, 0, 10, 10, 0.3f)});
    ASSERT_FALSE(r2.empty()) << "track should survive via Stage 2 low-conf match";
    EXPECT_EQ(r2[0].id, saved_id);
}

TEST(ByteTrackerTest, HighConfOnlyBehavesLikeSORTWhenNoLowConf) {
    ByteTracker bt;
    bt.min_hits(1).max_age(3)
      .high_conf_threshold(0.6f).low_conf_threshold(0.1f);

    SortTracker sort;
    sort.min_hits(1).max_age(3).iou_threshold(0.3f);

    for (int f = 0; f < 4; ++f) {
        float x = static_cast<float>(f) * 2.0f;
        auto d = std::vector<Detection>{make_det(x, 0, 10, 10, 0.9f)};
        auto r_bt   = bt.update(d);
        auto r_sort = sort.update(d);
        EXPECT_EQ(r_bt.size(), r_sort.size()) << "frame " << f;
    }
}

TEST(ByteTrackerTest, BelowLowThresholdIgnored) {
    ByteTracker tracker;
    tracker.min_hits(1).max_age(1)
           .high_conf_threshold(0.6f).low_conf_threshold(0.2f);

    tracker.update({make_det(0, 0, 10, 10, 0.9f)});
    // Detection with confidence below low threshold — should be ignored
    tracker.update({make_det(0, 0, 10, 10, 0.05f)});
    // Track should die (no match in either stage)
    auto r2 = tracker.update({});
    EXPECT_TRUE(r2.empty());
}

TEST(ByteTrackerTest, TwoStageMatchingUsesLowConf) {
    ByteTracker tracker;
    tracker.min_hits(1).max_age(0)
           .high_conf_threshold(0.6f).low_conf_threshold(0.1f);

    tracker.update({make_det(0, 0, 20, 20, 0.9f)});
    tracker.update({make_det(0, 0, 20, 20, 0.9f)});
    // Low-conf detection at same position — no high-conf dets so Stage 1 skips; Stage 2 must match
    auto r = tracker.update({make_det(0, 0, 20, 20, 0.3f)});
    ASSERT_FALSE(r.empty()) << "track should survive via Stage 2 low-conf match";
    EXPECT_EQ(r[0].id, 0);
    EXPECT_EQ(r[0].age, 0) << "age should be reset by correct_bbox in Stage 2";
}

TEST(ByteTrackerTest, ResetClearsState) {
    ByteTracker tracker;
    tracker.min_hits(1);
    tracker.update({make_det(0, 0, 10, 10, 0.9f)});
    tracker.update({make_det(0, 0, 10, 10, 0.9f)});
    tracker.reset();
    auto r = tracker.update({make_det(0, 0, 10, 10, 0.9f)});
    ASSERT_FALSE(r.empty());
    for (const auto& t : r) EXPECT_EQ(t.id, 0);
}

TEST(ByteTrackerTest, ThrowsOnInvalidSetterRange) {
    ByteTracker t;
    EXPECT_THROW(t.max_age(-1),                improc::ParameterError);
    EXPECT_THROW(t.min_hits(0),                improc::ParameterError);
    EXPECT_THROW(t.high_conf_threshold(-0.1f), improc::ParameterError);
    EXPECT_THROW(t.high_conf_threshold(1.1f),  improc::ParameterError);
    EXPECT_THROW(t.low_conf_threshold(-0.1f),  improc::ParameterError);
    EXPECT_THROW(t.low_conf_threshold(1.1f),   improc::ParameterError);
    EXPECT_NO_THROW(t.max_age(0));
    EXPECT_NO_THROW(t.min_hits(1));
    EXPECT_NO_THROW(t.high_conf_threshold(1.0f));
    EXPECT_NO_THROW(t.low_conf_threshold(0.0f));
}

TEST(ByteTrackerTest, ThrowsWhenLowThreshNotLessThanHigh) {
    // Cross-invariant checked at update() time to avoid order-dependency in setters
    ByteTracker t;
    t.high_conf_threshold(0.4f);
    t.low_conf_threshold(0.4f);  // equal — not strictly less
    EXPECT_THROW(t.update({}), improc::ParameterError);
}

TEST(ByteTrackerTest, EmptyDetectionsOnFreshTrackerReturnsEmpty) {
    ByteTracker tracker;
    EXPECT_TRUE(tracker.update({}).empty());
}
