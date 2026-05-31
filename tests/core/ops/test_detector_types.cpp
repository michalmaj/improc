// tests/core/ops/test_detector_types.cpp
#include <gtest/gtest.h>
#include "improc/core/ops/detector_types.hpp"

using namespace improc::core;

// ── MSERResult ────────────────────────────────────────────────────────────────

TEST(MSERResultTest, DefaultIsEmpty) {
    MSERResult r;
    EXPECT_TRUE(r.empty());
    EXPECT_EQ(r.size(), 0u);
}

TEST(MSERResultTest, SizeMatchesRegionCount) {
    MSERResult r;
    r.regions.push_back({{0,0},{1,1}});
    r.bboxes.push_back(cv::Rect{0,0,2,2});
    EXPECT_FALSE(r.empty());
    EXPECT_EQ(r.size(), 1u);
}

// ── LineSet ───────────────────────────────────────────────────────────────────

TEST(LineSetTest, DefaultIsEmpty) {
    LineSet ls;
    EXPECT_TRUE(ls.empty());
    EXPECT_EQ(ls.size(), 0u);
}

TEST(LineSetTest, SizeMatchesLineCount) {
    LineSet ls;
    ls.lines.push_back({0.f, 0.f, 100.f, 100.f});
    EXPECT_FALSE(ls.empty());
    EXPECT_EQ(ls.size(), 1u);
}

// ── QRResult ──────────────────────────────────────────────────────────────────

TEST(QRResultTest, DefaultIsEmpty) {
    QRResult r;
    EXPECT_TRUE(r.empty());
    EXPECT_EQ(r.size(), 0u);
}

TEST(QRResultTest, SizeMatchesDecodedCount) {
    QRResult r;
    r.decoded.push_back("hello");
    r.points.push_back(cv::Mat());
    EXPECT_FALSE(r.empty());
    EXPECT_EQ(r.size(), 1u);
}

// ── BarcodeResult ─────────────────────────────────────────────────────────────

TEST(BarcodeResultTest, DefaultIsEmpty) {
    BarcodeResult r;
    EXPECT_TRUE(r.empty());
    EXPECT_EQ(r.size(), 0u);
}

TEST(BarcodeResultTest, SizeMatchesDecodedCount) {
    BarcodeResult r;
    r.decoded.push_back("12345");
    r.types.push_back("EAN_13");
    r.bboxes.push_back(cv::RotatedRect{});
    EXPECT_FALSE(r.empty());
    EXPECT_EQ(r.size(), 1u);
    EXPECT_EQ(r.decoded[0], "12345");
    EXPECT_EQ(r.types[0], "EAN_13");
}

// ── FaceDetection ─────────────────────────────────────────────────────────────

TEST(FaceDetectionTest, FieldsAccessible) {
    FaceDetection fd;
    fd.bbox       = cv::Rect2f{10.f, 20.f, 100.f, 120.f};
    fd.confidence = 0.99f;
    fd.landmarks[0] = cv::Point2f{15.f, 30.f};

    EXPECT_FLOAT_EQ(fd.bbox.x, 10.f);
    EXPECT_FLOAT_EQ(fd.confidence, 0.99f);
    EXPECT_FLOAT_EQ(fd.landmarks[0].x, 15.f);
    EXPECT_EQ(fd.landmarks.size(), 5u);
}
