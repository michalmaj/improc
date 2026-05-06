// tests/core/ops/test_describe.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

static Image<Gray> make_textured() {
    cv::Mat m(200, 200, CV_8UC1, cv::Scalar(128));
    cv::rectangle(m, {10,  10}, {90,  90},  cv::Scalar(255), -1);
    cv::rectangle(m, {110, 10}, {190, 90},  cv::Scalar(0),   -1);
    cv::circle(m, {50,  150}, 40, cv::Scalar(200), -1);
    cv::circle(m, {150, 150}, 40, cv::Scalar(30),  -1);
    return Image<Gray>(m);
}

static Image<BGR> make_textured_bgr() {
    cv::Mat m(200, 200, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::rectangle(m, {10,  10}, {90,  90},  cv::Scalar(255, 255, 255), -1);
    cv::rectangle(m, {110, 10}, {190, 90},  cv::Scalar(0, 0, 0),       -1);
    cv::circle(m, {50,  150}, 40, cv::Scalar(200, 200, 200), -1);
    cv::circle(m, {150, 150}, 40, cv::Scalar(30, 30, 30),    -1);
    return Image<BGR>(m);
}

TEST(DescriptorSetTest, DefaultIsEmpty) {
    DescriptorSet ds;
    EXPECT_EQ(ds.size(), 0u);
    EXPECT_TRUE(ds.empty());
    EXPECT_TRUE(ds.descriptors.empty());
}

TEST(DescriptorSetTest, SizeMatchesKeypoints) {
    Image<Gray> src = make_textured();
    KeypointSet kps = src | DetectORB{};
    DescriptorSet ds = src | DescribeORB{kps};
    EXPECT_EQ(ds.size(), ds.keypoints.size());
}

TEST(DescribeORBTest, NonEmptyKeypointsGivesDescriptors) {
    Image<Gray> src = make_textured();
    KeypointSet kps = src | DetectORB{};
    DescriptorSet ds = src | DescribeORB{kps};
    EXPECT_FALSE(ds.empty());
    EXPECT_GT(ds.descriptors.rows, 0);
}

TEST(DescribeORBTest, DescriptorTypeIsCV8U) {
    Image<Gray> src = make_textured();
    KeypointSet kps = src | DetectORB{};
    DescriptorSet ds = src | DescribeORB{kps};
    EXPECT_EQ(ds.descriptors.type(), CV_8U);
}

TEST(DescribeORBTest, DescriptorCountMatchesKeypoints) {
    Image<Gray> src = make_textured();
    KeypointSet kps = src | DetectORB{};
    DescriptorSet ds = src | DescribeORB{kps};
    EXPECT_EQ(static_cast<std::size_t>(ds.descriptors.rows), ds.size());
}

TEST(DescribeORBTest, BGROverloadWorks) {
    Image<BGR>  src_bgr  = make_textured_bgr();
    Image<Gray> src_gray = make_textured();
    KeypointSet kps = src_gray | DetectORB{};
    DescriptorSet ds = src_bgr | DescribeORB{kps};
    EXPECT_FALSE(ds.empty());
}

TEST(DescribeORBTest, EmptyKeypointsGivesEmptyDescriptors) {
    Image<Gray> src = make_textured();
    KeypointSet empty_kps{};
    DescriptorSet ds = src | DescribeORB{empty_kps};
    EXPECT_TRUE(ds.empty());
    EXPECT_EQ(ds.descriptors.rows, 0);
}

TEST(DescribeSIFTTest, NonEmptyKeypointsGivesDescriptors) {
    Image<Gray> src = make_textured();
    KeypointSet kps = src | DetectSIFT{};
    DescriptorSet ds = src | DescribeSIFT{kps};
    EXPECT_FALSE(ds.empty());
}

TEST(DescribeSIFTTest, DescriptorTypeIsCV32F) {
    Image<Gray> src = make_textured();
    KeypointSet kps = src | DetectSIFT{};
    DescriptorSet ds = src | DescribeSIFT{kps};
    EXPECT_EQ(ds.descriptors.type(), CV_32F);
}

TEST(DescribeSIFTTest, DescriptorCountMatchesKeypoints) {
    Image<Gray> src = make_textured();
    KeypointSet kps = src | DetectSIFT{};
    DescriptorSet ds = src | DescribeSIFT{kps};
    EXPECT_EQ(static_cast<std::size_t>(ds.descriptors.rows), ds.size());
}

TEST(DescribeSIFTTest, BGROverloadWorks) {
    Image<BGR>  src_bgr  = make_textured_bgr();
    Image<Gray> src_gray = make_textured();
    KeypointSet kps = src_gray | DetectSIFT{};
    DescriptorSet ds = src_bgr | DescribeSIFT{kps};
    EXPECT_FALSE(ds.empty());
}

TEST(DescribeAKAZETest, NonEmptyKeypointsGivesDescriptors) {
    Image<Gray> src = make_textured();
    KeypointSet kps = src | DetectAKAZE{};
    DescriptorSet ds = src | DescribeAKAZE{kps};
    EXPECT_FALSE(ds.empty());
}

TEST(DescribeAKAZETest, DescriptorTypeIsCV8U) {
    Image<Gray> src = make_textured();
    KeypointSet kps = src | DetectAKAZE{};
    DescriptorSet ds = src | DescribeAKAZE{kps};
    EXPECT_EQ(ds.descriptors.type(), CV_8U);
}

TEST(DescribeAKAZETest, DescriptorCountMatchesKeypoints) {
    Image<Gray> src = make_textured();
    KeypointSet kps = src | DetectAKAZE{};
    DescriptorSet ds = src | DescribeAKAZE{kps};
    EXPECT_EQ(static_cast<std::size_t>(ds.descriptors.rows), ds.size());
}

TEST(DescribeAKAZETest, BGROverloadWorks) {
    Image<BGR>  src_bgr  = make_textured_bgr();
    Image<Gray> src_gray = make_textured();
    KeypointSet kps = src_gray | DetectAKAZE{};
    DescriptorSet ds = src_bgr | DescribeAKAZE{kps};
    EXPECT_FALSE(ds.empty());
}

TEST(DescribeSIFTTest, EmptyKeypointsGivesEmptyDescriptors) {
    Image<Gray> src = make_textured();
    KeypointSet empty_kps{};
    DescriptorSet ds = src | DescribeSIFT{empty_kps};
    EXPECT_TRUE(ds.empty());
    EXPECT_EQ(ds.descriptors.rows, 0);
}

TEST(DescribeAKAZETest, EmptyKeypointsGivesEmptyDescriptors) {
    Image<Gray> src = make_textured();
    KeypointSet empty_kps{};
    DescriptorSet ds = src | DescribeAKAZE{empty_kps};
    EXPECT_TRUE(ds.empty());
    EXPECT_EQ(ds.descriptors.rows, 0);
}
