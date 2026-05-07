// tests/core/ops/test_draw_matches.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

static Image<Gray> make_gray() {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(128));
    cv::rectangle(m, {10, 10}, {90, 90},   cv::Scalar(255), -1);
    cv::rectangle(m, {110, 10}, {190, 90}, cv::Scalar(0),   -1);
    cv::circle(m, {50,  150}, 40, cv::Scalar(200), -1);
    cv::circle(m, {150, 150}, 40, cv::Scalar(30),  -1);
    return Image<Gray>(m);
}

static Image<BGR> make_bgr() {
    cv::Mat m(200, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::rectangle(m, {10, 10}, {90, 90},   cv::Scalar(255, 255, 255), -1);
    cv::rectangle(m, {110, 10}, {190, 90}, cv::Scalar(0, 0, 0),       -1);
    return Image<BGR>(m);
}

static KeypointSet make_known_kps() {
    KeypointSet ks;
    ks.keypoints.emplace_back(cv::Point2f(100.0f, 100.0f), 20.0f);
    ks.keypoints.emplace_back(cv::Point2f( 50.0f,  50.0f), 10.0f);
    return ks;
}

// ── DrawKeypoints ─────────────────────────────────────────────────────────────

TEST(DrawKeypointsTest, GrayInputReturnsBGR) {
    Image<Gray> src = make_gray();
    Image<BGR> out = src | DrawKeypoints{KeypointSet{}};
    EXPECT_EQ(out.mat().channels(), 3);
    EXPECT_EQ(out.mat().type(), CV_8UC3);
}

TEST(DrawKeypointsTest, BGRInputReturnsBGR) {
    Image<BGR> src = make_bgr();
    Image<BGR> out = src | DrawKeypoints{KeypointSet{}};
    EXPECT_EQ(out.mat().channels(), 3);
    EXPECT_EQ(out.mat().type(), CV_8UC3);
}

TEST(DrawKeypointsTest, OutputSizeMatchesInput) {
    Image<Gray> src = make_gray();
    Image<BGR> out = src | DrawKeypoints{make_known_kps()};
    EXPECT_EQ(out.mat().rows, src.mat().rows);
    EXPECT_EQ(out.mat().cols, src.mat().cols);
}

TEST(DrawKeypointsTest, EmptyKeypointsProducesValidBGR) {
    Image<Gray> src = make_gray();
    Image<BGR> out = src | DrawKeypoints{KeypointSet{}};
    EXPECT_FALSE(out.mat().empty());
    EXPECT_EQ(out.mat().type(), CV_8UC3);
}

TEST(DrawKeypointsTest, NonEmptyKeypointsModifiesOutput) {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(0));
    Image<Gray> blank(m);
    Image<BGR> out_empty = blank | DrawKeypoints{KeypointSet{}};
    Image<BGR> out_kp    = blank | DrawKeypoints{make_known_kps()};
    cv::Mat diff;
    cv::absdiff(out_empty.mat(), out_kp.mat(), diff);
    cv::Mat diff_gray;
    cv::cvtColor(diff, diff_gray, cv::COLOR_BGR2GRAY);
    EXPECT_GT(cv::countNonZero(diff_gray), 0);
}

// ── DrawMatches ───────────────────────────────────────────────────────────────

TEST(DrawMatchesTest, OutputIsBGRFormat) {
    Image<BGR> img1 = make_bgr();
    Image<BGR> img2 = make_bgr();
    KeypointSet kps;
    MatchSet ms;
    Image<BGR> out = DrawMatches{img1, kps, img2, kps, ms}();
    EXPECT_EQ(out.mat().type(), CV_8UC3);
    EXPECT_FALSE(out.mat().empty());
}

TEST(DrawMatchesTest, OutputWidthIsSumOfInputWidths) {
    Image<BGR> img1 = make_bgr();
    Image<BGR> img2 = make_bgr();
    KeypointSet kps;
    MatchSet ms;
    Image<BGR> out = DrawMatches{img1, kps, img2, kps, ms}();
    EXPECT_EQ(out.mat().cols, img1.mat().cols + img2.mat().cols);
}

TEST(DrawMatchesTest, NonEmptyMatchesProducesOutput) {
    Image<BGR> img1 = make_bgr();
    Image<BGR> img2 = make_bgr();
    Image<Gray> gray1 = img1 | ToGray{};
    Image<Gray> gray2 = img2 | ToGray{};
    KeypointSet kps1 = gray1 | DetectORB{};
    DescriptorSet desc1 = gray1 | DescribeORB{kps1};
    KeypointSet kps2 = gray2 | DetectORB{};
    DescriptorSet desc2 = gray2 | DescribeORB{kps2};
    MatchSet ms = MatchBF{desc1, desc2}();
    Image<BGR> out = DrawMatches{img1, kps1, img2, kps2, ms}();
    EXPECT_FALSE(out.mat().empty());
    EXPECT_EQ(out.mat().cols, img1.mat().cols + img2.mat().cols);
}
