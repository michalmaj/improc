// tests/core/ops/test_channels.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(SplitChannelsTest, BGRSplitsToThreeGrays) {
    cv::Mat m(50, 50, CV_8UC3, cv::Scalar(10, 20, 30));
    Image<BGR> img(m);
    auto [b, g, r] = SplitChannels{}(img);
    EXPECT_EQ(b.rows(), 50);
    EXPECT_EQ(b.cols(), 50);
    EXPECT_EQ(b.mat().at<uchar>(0, 0), 10u);  // B channel
    EXPECT_EQ(g.mat().at<uchar>(0, 0), 20u);  // G channel
    EXPECT_EQ(r.mat().at<uchar>(0, 0), 30u);  // R channel
}

TEST(SplitChannelsTest, BGRASplitsToFourGrays) {
    cv::Mat m(50, 50, CV_8UC4, cv::Scalar(10, 20, 30, 255));
    Image<BGRA> img(m);
    auto [b, g, r, a] = SplitChannels{}(img);
    EXPECT_EQ(a.mat().at<uchar>(0, 0), 255u);
}

TEST(MergeChannelsTest, ThreeGraysToBGR) {
    Image<Gray> b(cv::Mat(50, 50, CV_8UC1, cv::Scalar(10)));
    Image<Gray> g(cv::Mat(50, 50, CV_8UC1, cv::Scalar(20)));
    Image<Gray> r(cv::Mat(50, 50, CV_8UC1, cv::Scalar(30)));
    auto bgr = MergeChannels{}(b, g, r);
    EXPECT_EQ(bgr.mat().at<cv::Vec3b>(0, 0)[0], 10u);
    EXPECT_EQ(bgr.mat().at<cv::Vec3b>(0, 0)[1], 20u);
    EXPECT_EQ(bgr.mat().at<cv::Vec3b>(0, 0)[2], 30u);
}

TEST(MergeChannelsTest, FourGraysToBGRA) {
    Image<Gray> b(cv::Mat(50, 50, CV_8UC1, cv::Scalar(10)));
    Image<Gray> g(cv::Mat(50, 50, CV_8UC1, cv::Scalar(20)));
    Image<Gray> r(cv::Mat(50, 50, CV_8UC1, cv::Scalar(30)));
    Image<Gray> a(cv::Mat(50, 50, CV_8UC1, cv::Scalar(255)));
    auto bgra = MergeChannels{}(b, g, r, a);
    EXPECT_EQ(bgra.mat().channels(), 4);
    EXPECT_EQ(bgra.mat().at<cv::Vec4b>(0, 0)[0], 10u);
    EXPECT_EQ(bgra.mat().at<cv::Vec4b>(0, 0)[3], 255u);
}

TEST(MergeChannelsTest, SplitRoundtrip) {
    cv::Mat m(50, 50, CV_8UC3);
    cv::RNG rng(42);
    rng.fill(m, cv::RNG::UNIFORM, 0, 256);
    Image<BGR> original(m.clone());
    auto [b, g, r] = SplitChannels{}(original);
    auto reconstructed = MergeChannels{}(b, g, r);
    cv::Mat diff;
    cv::absdiff(original.mat(), reconstructed.mat(), diff);
    EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(MergeChannelsTest, MismatchedSizesThrow) {
    Image<Gray> b(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    Image<Gray> g(cv::Mat(60, 60, CV_8UC1, cv::Scalar(0)));
    Image<Gray> r(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    EXPECT_THROW(MergeChannels{}(b, g, r), std::invalid_argument);
}

TEST(MergeChannelsTest, MismatchedAlphaThrows) {
    Image<Gray> b(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    Image<Gray> g(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    Image<Gray> r(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    Image<Gray> a(cv::Mat(60, 60, CV_8UC1, cv::Scalar(255)));
    EXPECT_THROW(MergeChannels{}(b, g, r, a), std::invalid_argument);
}
