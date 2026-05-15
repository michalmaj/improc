// tests/ml/test_tracking_eval.cpp
#include <gtest/gtest.h>
#include "improc/ml/tracking/tracking_eval.hpp"

using namespace improc::ml;

static Track make_track(int id, float x, float y, float w, float h) {
    Track t;
    t.id = id;
    t.bbox = BBox{cv::Rect2f(x, y, w, h), 0, ""};
    t.is_confirmed = true;
    return t;
}

static TrackGT make_gt(int id, float x, float y, float w, float h) {
    return TrackGT{id, BBox{cv::Rect2f(x, y, w, h), 0, ""}};
}

TEST(TrackingEvalTest, PerfectTracking) {
    TrackingEval eval;
    for (int f = 0; f < 3; ++f)
        eval.update({make_track(0, 0,0,10,10)}, {make_gt(0, 0,0,10,10)});
    auto m = eval.compute();
    EXPECT_NEAR(m.MOTA, 1.0f, 1e-4f);
    EXPECT_NEAR(m.MOTP, 1.0f, 1e-4f);
    EXPECT_EQ(m.IDSW, 0);
    EXPECT_EQ(m.FP,   0);
    EXPECT_EQ(m.FN,   0);
}

TEST(TrackingEvalTest, AllMissed) {
    TrackingEval eval;
    eval.update({}, {make_gt(0, 0,0,10,10), make_gt(1, 50,0,10,10)});
    auto m = eval.compute();
    EXPECT_EQ(m.FN, 2);
    EXPECT_FLOAT_EQ(m.MOTA, 1.0f - 2.0f / 2.0f);  // = 0.0
}

TEST(TrackingEvalTest, IdSwitch) {
    TrackingEval eval;
    eval.update({make_track(0, 0,0,10,10)}, {make_gt(0, 0,0,10,10)});
    eval.update({make_track(1, 0,0,10,10)}, {make_gt(0, 0,0,10,10)});
    auto m = eval.compute();
    EXPECT_EQ(m.IDSW, 1);
}

TEST(TrackingEvalTest, MOTPComputation) {
    // GT=[0,0,10,10] pred=[2,2,10,10]
    // intersection: [2,2,8,8] = 64; union: 100+100-64=136; IoU = 64/136 ~0.47
    // Use threshold below the IoU so the pair is matched.
    TrackingEval eval;
    eval.iou_threshold(0.3f);
    eval.update({make_track(0, 2,2,10,10)}, {make_gt(0, 0,0,10,10)});
    auto m = eval.compute();
    EXPECT_NEAR(m.MOTP, 64.0f / 136.0f, 1e-3f);
}

TEST(TrackingEvalTest, IDF1PerfectIds) {
    TrackingEval eval;
    for (int f = 0; f < 5; ++f)
        eval.update({make_track(0, 0,0,10,10)}, {make_gt(0, 0,0,10,10)});
    auto m = eval.compute();
    EXPECT_NEAR(m.IDF1, 1.0f, 1e-4f);
}

TEST(TrackingEvalTest, IDF1IdSwitchReducesScore) {
    TrackingEval eval;
    eval.update({make_track(0, 0,0,10,10)}, {make_gt(0, 0,0,10,10)});
    eval.update({make_track(1, 0,0,10,10)}, {make_gt(0, 0,0,10,10)});
    auto m = eval.compute();
    EXPECT_LT(m.IDF1, 1.0f);
}

TEST(TrackingEvalTest, ResetClearsState) {
    TrackingEval eval;
    eval.update({make_track(0, 0,0,10,10)}, {make_gt(0, 0,0,10,10)});
    eval.reset();
    auto m = eval.compute();
    EXPECT_FLOAT_EQ(m.MOTA, 0.0f);
    EXPECT_EQ(m.FP, 0);
    EXPECT_EQ(m.FN, 0);
}
