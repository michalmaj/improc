// tests/core/ops/test_to_bgr.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// ── Gray → BGR ───────────────────────────────────────────────────────────────

TEST(ToBGRTest, GrayToBGRProducesThreeChannels) {
    cv::Mat m(10, 10, CV_8UC1, cv::Scalar(128));
    Image<Gray> gray(m);
    Image<BGR> result = gray | ToBGR{};
    EXPECT_EQ(result.mat().channels(), 3);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
}

TEST(ToBGRTest, GrayToBGRReplicatesValueAcrossChannels) {
    cv::Mat m(1, 1, CV_8UC1, cv::Scalar(100));
    Image<Gray> gray(m);
    Image<BGR> result = gray | ToBGR{};
    cv::Vec3b pixel = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_EQ(pixel[0], 100);
    EXPECT_EQ(pixel[1], 100);
    EXPECT_EQ(pixel[2], 100);
}

TEST(ToBGRTest, GrayToBGROutputTypeIsCV_8UC3) {
    cv::Mat m(4, 4, CV_8UC1, cv::Scalar(200));
    Image<Gray> gray(m);
    Image<BGR> result = gray | ToBGR{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

// ── HSV → BGR ────────────────────────────────────────────────────────────────

TEST(ToBGRTest, HSVToBGROutputTypeIsCV_8UC3) {
    cv::Mat m(4, 4, CV_8UC3, cv::Scalar(0, 255, 255));
    Image<HSV> hsv(m);
    Image<BGR> result = hsv | ToBGR{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 4);
}

TEST(ToBGRTest, HSVToBGRMatchesFreeFunction) {
    cv::Mat m(4, 4, CV_8UC3, cv::Scalar(30, 200, 180));
    Image<HSV> hsv(m);
    Image<BGR> via_op       = hsv | ToBGR{};
    Image<BGR> via_function = convert<BGR, HSV>(hsv);
    EXPECT_EQ(cv::norm(via_op.mat(), via_function.mat(), cv::NORM_INF), 0.0);
}

// ── LAB → BGR ────────────────────────────────────────────────────────────────

TEST(ToBGRTest, LABToBGROutputTypeIsCV_8UC3) {
    // L=100 (white), a=128 (neutral), b=128 (neutral) in 8-bit LAB
    cv::Mat m(4, 4, CV_8UC3, cv::Scalar(255, 128, 128));
    Image<LAB> lab(m);
    Image<BGR> result = lab | ToBGR{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 4);
}

TEST(ToBGRTest, LABToBGRMatchesFreeFunction) {
    cv::Mat m(4, 4, CV_8UC3, cv::Scalar(100, 128, 128));
    Image<LAB> lab(m);
    Image<BGR> via_op       = lab | ToBGR{};
    Image<BGR> via_function = convert<BGR, LAB>(lab);
    EXPECT_EQ(cv::norm(via_op.mat(), via_function.mat(), cv::NORM_INF), 0.0);
}

// ── YCrCb → BGR ──────────────────────────────────────────────────────────────

TEST(ToBGRTest, YCrCbToBGROutputTypeIsCV_8UC3) {
    // Y=128, Cr=128, Cb=128 (neutral gray in YCrCb)
    cv::Mat m(4, 4, CV_8UC3, cv::Scalar(128, 128, 128));
    Image<YCrCb> ycrcb(m);
    Image<BGR> result = ycrcb | ToBGR{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.rows(), 4);
    EXPECT_EQ(result.cols(), 4);
}

TEST(ToBGRTest, YCrCbToBGRMatchesFreeFunction) {
    cv::Mat m(4, 4, CV_8UC3, cv::Scalar(128, 128, 128));
    Image<YCrCb> ycrcb(m);
    Image<BGR> via_op       = ycrcb | ToBGR{};
    Image<BGR> via_function = convert<BGR, YCrCb>(ycrcb);
    EXPECT_EQ(cv::norm(via_op.mat(), via_function.mat(), cv::NORM_INF), 0.0);
}
