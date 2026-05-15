// tests/ml/test_detection_eval.cpp
#include <gtest/gtest.h>
#include <vector>
#include "improc/ml/eval/detection.hpp"

using namespace improc::ml;

static BBox make_box(float x, float y, float w, float h,
                     std::string label = "cat", int class_id = 0) {
    return BBox{cv::Rect2f(x, y, w, h), class_id, std::move(label)};
}

// ── iou ──────────────────────────────────────────────────────────────────────

TEST(IouTest, PerfectOverlap) {
    auto b = make_box(0, 0, 10, 10);
    EXPECT_FLOAT_EQ(iou(b, b), 1.0f);
}

TEST(IouTest, NoOverlap) {
    auto a = make_box(0,  0, 10, 10);
    auto b = make_box(20, 20, 10, 10);
    EXPECT_FLOAT_EQ(iou(a, b), 0.0f);
}

TEST(IouTest, PartialOverlap) {
    // a=[0,0,10,10], b=[5,5,10,10]
    // intersection: [5,5,10,10]=5×5=25; union=100+100-25=175
    auto a = make_box(0, 0, 10, 10);
    auto b = make_box(5, 5, 10, 10);
    EXPECT_NEAR(iou(a, b), 25.0f / 175.0f, 1e-5f);
}

TEST(IouTest, ZeroAreaBox) {
    auto a = make_box(0, 0, 0, 0);
    auto b = make_box(0, 0, 10, 10);
    EXPECT_FLOAT_EQ(iou(a, b), 0.0f);
}

// ── average_precision ────────────────────────────────────────────────────────

TEST(AveragePrecisionTest, PerfectCurve) {
    std::vector<float> r = {0.0f, 0.5f, 1.0f};
    std::vector<float> p = {1.0f, 1.0f, 1.0f};
    EXPECT_NEAR(average_precision(r, p), 1.0f, 1e-4f);
}

TEST(AveragePrecisionTest, ZeroCurve) {
    std::vector<float> r = {0.0f, 0.5f, 1.0f};
    std::vector<float> p = {0.0f, 0.0f, 0.0f};
    EXPECT_FLOAT_EQ(average_precision(r, p), 0.0f);
}
