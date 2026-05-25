// tests/core/ops/test_detectors.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Image<Gray> make_blank_gray() {
    return Image<Gray>(cv::Mat(200, 200, CV_8UC1, cv::Scalar(255)));
}

static Image<Gray> make_corner_scene() {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(255));
    // Draw a distinct cross/checkerboard pattern with high contrast
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            if ((i + j) % 2 == 0)
                cv::rectangle(m, {i*20, j*20}, {i*20+19, j*20+19}, cv::Scalar(0), -1);
        }
    }
    return Image<Gray>(m);
}

static Image<Gray> make_circles_gray() {
    cv::Mat m(300, 300, CV_8UC1, cv::Scalar(255));
    cv::circle(m, {75,  150}, 30, cv::Scalar(0), -1);
    cv::circle(m, {150, 150}, 30, cv::Scalar(0), -1);
    cv::circle(m, {225, 150}, 30, cv::Scalar(0), -1);
    return Image<Gray>(m);
}

static Image<Gray> make_nested_rect_gray() {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(255));
    cv::rectangle(m, {40, 40}, {160, 160}, cv::Scalar(0), -1);
    cv::rectangle(m, {70, 70}, {130, 130}, cv::Scalar(200), -1);
    return Image<Gray>(m);
}

static Image<Gray> make_diagonal_line_gray() {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(255));
    cv::line(m, {20, 20}, {180, 180}, cv::Scalar(0), 3);
    return Image<Gray>(m);
}

static Image<BGR> make_blank_bgr() {
    return Image<BGR>(cv::Mat(200, 200, CV_8UC3, cv::Scalar(255, 255, 255)));
}

// ── DetectFAST ────────────────────────────────────────────────────────────────

TEST(DetectFASTTest, FindsCornersInCornerScene) {
    auto img = make_circles_gray();
    auto result = DetectFAST{}.threshold(30)(img);
    EXPECT_GT(result.size(), 0u);
}

TEST(DetectFASTTest, BlankImageNoKeypoints) {
    auto img = make_blank_gray();
    auto result = DetectFAST{}(img);
    EXPECT_TRUE(result.empty());
}

TEST(DetectFASTTest, ThresholdReducesCount) {
    auto img = make_corner_scene();
    auto low  = DetectFAST{}.threshold(5)(img);
    auto high = DetectFAST{}.threshold(200)(img);
    EXPECT_GE(low.size(), high.size());
}

// ── DetectBlob ───────────────────────────────────────────────────────────────

TEST(DetectBlobTest, DetectsThreeCircles) {
    auto img = make_circles_gray();
    auto result = DetectBlob{}(img);
    EXPECT_EQ(result.size(), 3u);
}

TEST(DetectBlobTest, BlankImageNoBlobs) {
    auto img = make_blank_gray();
    auto result = DetectBlob{}(img);
    EXPECT_TRUE(result.empty());
}

TEST(DetectBlobTest, KeypointsWithinCircleRegion) {
    auto img = make_circles_gray();
    auto result = DetectBlob{}(img);
    ASSERT_EQ(result.size(), 3u);
    for (const auto& kp : result.keypoints) {
        EXPECT_GE(kp.pt.x, 40.f);
        EXPECT_LE(kp.pt.x, 260.f);
        EXPECT_NEAR(kp.pt.y, 150.f, 20.f);
    }
}

// ── DetectMSER ───────────────────────────────────────────────────────────────

TEST(DetectMSERTest, FindsRegionInNestedRects) {
    auto img = make_nested_rect_gray();
    auto result = DetectMSER{}(img);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.regions.size(), result.bboxes.size());
}

TEST(DetectMSERTest, BlankImageNoRegions) {
    auto img = make_blank_gray();
    auto result = DetectMSER{}(img);
    EXPECT_TRUE(result.empty());
}

TEST(DetectMSERTest, SizeAndBboxesConsistent) {
    auto img = make_nested_rect_gray();
    auto result = DetectMSER{}(img);
    EXPECT_EQ(result.size(), result.bboxes.size());
}

// ── DetectLines ──────────────────────────────────────────────────────────────

TEST(DetectLinesTest, DetectsDiagonalLine) {
    auto img = make_diagonal_line_gray();
    auto result = DetectLines{}(img);
    EXPECT_FALSE(result.empty());
}

TEST(DetectLinesTest, BlankImageNoLines) {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(255));
    Image<Gray> img(m);
    auto result = DetectLines{}(img);
    EXPECT_TRUE(result.empty());
}

TEST(DetectLinesTest, DetectedAngleMatchesDrawnAngle) {
    auto img = make_diagonal_line_gray();
    auto result = DetectLines{}(img);
    ASSERT_FALSE(result.empty());
    const auto& l = result.lines[0];
    double angle_rad = std::atan2(l[3] - l[1], l[2] - l[0]);
    double angle_deg = std::abs(angle_rad * 180.0 / CV_PI);
    // 45° diagonal — allow wide tolerance for LSD sub-pixel fitting
    EXPECT_NEAR(angle_deg, 45.0, 15.0);
}

// ── DetectQR ─────────────────────────────────────────────────────────────────

namespace {
Image<BGR> make_qr_scene(const std::string& text) {
    auto encoder = cv::QRCodeEncoder::create();
    cv::Mat qr_gray;
    encoder->encode(text, qr_gray);
    cv::Mat large;
    cv::resize(qr_gray, large, {200, 200}, 0.0, 0.0, cv::INTER_NEAREST);
    cv::Mat bgr;
    cv::cvtColor(large, bgr, cv::COLOR_GRAY2BGR);
    return Image<BGR>(bgr);
}
} // namespace

TEST(DetectQRTest, DecodesGeneratedCode) {
    auto img = make_qr_scene("HELLO");
    auto result = DetectQR{}(img);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result.decoded[0], "HELLO");
}

TEST(DetectQRTest, BlankImageNoQR) {
    auto img = make_blank_bgr();
    auto result = DetectQR{}(img);
    EXPECT_TRUE(result.empty());
}

TEST(DetectQRTest, PointsCountMatchesDecoded) {
    auto img = make_qr_scene("TEST");
    auto result = DetectQR{}(img);
    EXPECT_EQ(result.decoded.size(), result.points.size());
}

// ── DetectBarcode ─────────────────────────────────────────────────────────────

TEST(DetectBarcodeTest, BlankImageNoBarcode) {
    auto img = make_blank_bgr();
    auto result = DetectBarcode{}(img);
    EXPECT_TRUE(result.empty());
}

TEST(DetectBarcodeTest, ResultSizesConsistent) {
    auto img = make_blank_bgr();
    auto result = DetectBarcode{}(img);
    EXPECT_EQ(result.decoded.size(), result.types.size());
    EXPECT_EQ(result.decoded.size(), result.bboxes.size());
}
