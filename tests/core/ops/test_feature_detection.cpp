// tests/core/ops/test_feature_detection.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;

static Image<Gray> make_textured() {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(128));
    cv::rectangle(m, {10,  10},  {90,  90},  cv::Scalar(255), -1);
    cv::rectangle(m, {110, 10},  {190, 90},  cv::Scalar(0),   -1);
    cv::circle(m, {50,  150}, 40, cv::Scalar(200), -1);
    cv::circle(m, {150, 150}, 40, cv::Scalar(30),  -1);
    return Image<Gray>(m);
}

static Image<Gray> make_blank() {
    return Image<Gray>(cv::Mat(200, 200, CV_8UC1, cv::Scalar(128)));
}

// ── KeypointSet ───────────────────────────────────────────────────────────────

TEST(KeypointSetTest, DefaultIsEmpty) {
    KeypointSet ks;
    EXPECT_EQ(ks.size(), 0u);
    EXPECT_TRUE(ks.empty());
    EXPECT_TRUE(ks.keypoints.empty());
}

TEST(KeypointSetTest, SizeMatchesKeypoints) {
    Image<Gray> src = make_textured();
    KeypointSet ks = src | DetectORB{};
    EXPECT_EQ(ks.size(), ks.keypoints.size());
}

// ── DetectORB ─────────────────────────────────────────────────────────────────

TEST(DetectORBTest, FindsKeypointsInTexturedImage) {
    Image<Gray> src = make_textured();
    KeypointSet ks = src | DetectORB{};
    EXPECT_GT(ks.size(), 0u);
}

TEST(DetectORBTest, BlankImageHasNoKeypoints) {
    Image<Gray> src = make_blank();
    KeypointSet ks = src | DetectORB{};
    EXPECT_EQ(ks.size(), 0u);
}

TEST(DetectORBTest, MaxFeaturesLimitsCount) {
    Image<Gray> src = make_textured();
    KeypointSet ks = src | DetectORB{}.max_features(10);
    EXPECT_LE(ks.size(), 10u);
}

// ── DetectSIFT ────────────────────────────────────────────────────────────────

TEST(DetectSIFTTest, FindsKeypointsInTexturedImage) {
    Image<Gray> src = make_textured();
    KeypointSet ks = src | DetectSIFT{};
    EXPECT_GT(ks.size(), 0u);
}

TEST(DetectSIFTTest, BlankImageHasNoKeypoints) {
    Image<Gray> src = make_blank();
    KeypointSet ks = src | DetectSIFT{};
    EXPECT_EQ(ks.size(), 0u);
}

TEST(DetectSIFTTest, MaxFeaturesLimitsCount) {
    Image<Gray> src = make_textured();
    KeypointSet ks = src | DetectSIFT{}.max_features(5);
    EXPECT_LE(ks.size(), 5u);
}

// ── DetectAKAZE ───────────────────────────────────────────────────────────────

TEST(DetectAKAZETest, FindsKeypointsInTexturedImage) {
    Image<Gray> src = make_textured();
    KeypointSet ks = src | DetectAKAZE{};
    EXPECT_GT(ks.size(), 0u);
}

TEST(DetectAKAZETest, BlankImageHasNoKeypoints) {
    Image<Gray> src = make_blank();
    KeypointSet ks = src | DetectAKAZE{};
    EXPECT_EQ(ks.size(), 0u);
}

TEST(DetectAKAZETest, HigherThresholdFewerKeypoints) {
    Image<Gray> src = make_textured();
    KeypointSet loose  = src | DetectAKAZE{}.threshold(0.0001f);
    KeypointSet strict = src | DetectAKAZE{}.threshold(0.01f);
    EXPECT_GE(loose.size(), strict.size());
}

// ── GoodFeaturesToTrack ───────────────────────────────────────────────────────

static Image<Gray> make_chessboard() {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(255));
    for (int r = 0; r < 200; r += 20)
        for (int c = 0; c < 200; c += 20)
            if (((r / 20) + (c / 20)) % 2 == 0)
                cv::rectangle(m, {c, r}, {c + 19, r + 19}, cv::Scalar(0), -1);
    return Image<Gray>(m);
}

TEST(GoodFeaturesToTrackTest, DefaultConstruction) {
    EXPECT_NO_THROW(GoodFeaturesToTrack{});
}

TEST(GoodFeaturesToTrackTest, FluentSetterReturnsThis) {
    GoodFeaturesToTrack op;
    EXPECT_EQ(&op.max_corners(50).quality_level(0.05).min_distance(5.0).use_harris(true), &op);
}

TEST(GoodFeaturesToTrackTest, DetectsCornersInChessboard) {
    auto corners = GoodFeaturesToTrack{}.quality_level(0.01).min_distance(5.0)(make_chessboard());
    EXPECT_GE(corners.size(), 1u);
}

TEST(GoodFeaturesToTrackTest, BlankImageReturnsEmpty) {
    Image<Gray> img(cv::Mat(200, 200, CV_8UC1, cv::Scalar(128)));
    EXPECT_TRUE(GoodFeaturesToTrack{}(img).empty());
}

TEST(GoodFeaturesToTrackTest, QualityLevelZeroThrows) {
    EXPECT_THROW(GoodFeaturesToTrack{}.quality_level(0.0), improc::ParameterError);
}
TEST(GoodFeaturesToTrackTest, QualityLevelNegativeThrows) {
    EXPECT_THROW(GoodFeaturesToTrack{}.quality_level(-0.1), improc::ParameterError);
}
TEST(GoodFeaturesToTrackTest, MinDistanceNegativeThrows) {
    EXPECT_THROW(GoodFeaturesToTrack{}.min_distance(-1.0), improc::ParameterError);
}
