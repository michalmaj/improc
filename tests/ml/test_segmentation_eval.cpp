// tests/ml/test_segmentation_eval.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/ml/eval/segmentation.hpp"

using namespace improc::core;
using namespace improc::ml;

// ── pixel_iou ────────────────────────────────────────────────────────────────

TEST(PixelIouTest, PerfectMask) {
    cv::Mat m(8, 8, CV_8U, cv::Scalar(1));
    Image<Gray> mask(m);
    EXPECT_FLOAT_EQ(pixel_iou(mask, mask, 1), 1.0f);
}

TEST(PixelIouTest, NoOverlap) {
    cv::Mat pred_m(8, 8, CV_8U, cv::Scalar(1));
    cv::Mat gt_m(8, 8, CV_8U, cv::Scalar(2));
    EXPECT_FLOAT_EQ(pixel_iou(Image<Gray>(pred_m), Image<Gray>(gt_m), 1), 0.0f);
}

TEST(PixelIouTest, PartialOverlap) {
    // 4×4 mask; pred has class 1 in row 0+1, gt has class 1 in row 0 only
    // TP=4, FP=4, FN=0 → IoU = 4/(4+4+0) = 0.5
    cv::Mat pred_m(4, 4, CV_8U, cv::Scalar(0));
    cv::Mat gt_m(4, 4, CV_8U, cv::Scalar(0));
    pred_m.row(0).setTo(1);
    pred_m.row(1).setTo(1);
    gt_m.row(0).setTo(1);
    EXPECT_NEAR(pixel_iou(Image<Gray>(pred_m), Image<Gray>(gt_m), 1), 4.0f / 8.0f, 1e-5f);
}

TEST(PixelIouTest, VoidPixelsIgnored) {
    // gt void (255) pixels should not count
    cv::Mat pred_m(4, 4, CV_8U, cv::Scalar(1));
    cv::Mat gt_m(4, 4, CV_8U, cv::Scalar(255));
    EXPECT_FLOAT_EQ(pixel_iou(Image<Gray>(pred_m), Image<Gray>(gt_m), 1), 0.0f);
}

TEST(PixelIouTest, DimensionMismatchThrows) {
    cv::Mat pred_m(4, 4, CV_8U, cv::Scalar(1));
    cv::Mat gt_m(8, 8, CV_8U, cv::Scalar(1));
    EXPECT_THROW(pixel_iou(Image<Gray>(pred_m), Image<Gray>(gt_m), 1),
                 std::invalid_argument);
}

// ── dice ─────────────────────────────────────────────────────────────────────

TEST(DiceTest, PerfectMask) {
    cv::Mat m(8, 8, CV_8U, cv::Scalar(1));
    Image<Gray> mask(m);
    EXPECT_FLOAT_EQ(dice(mask, mask, 1), 1.0f);
}

TEST(DiceTest, SymmetricProperty) {
    cv::Mat a_m(4, 4, CV_8U, cv::Scalar(0));
    cv::Mat b_m(4, 4, CV_8U, cv::Scalar(0));
    a_m.row(0).setTo(1);
    b_m.row(1).setTo(1);
    Image<Gray> a(a_m), b(b_m);
    EXPECT_NEAR(dice(a, b, 1), dice(b, a, 1), 1e-6f);
}
