// tests/core/ops/test_flood_fill.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/flood_fill.hpp"

using namespace improc::core;

TEST(FloodFillTest, DefaultConstruction) {
    EXPECT_NO_THROW(FloodFill{});
}

TEST(FloodFillTest, FluentSetterReturnsThis) {
    FloodFill op;
    EXPECT_EQ(&op.lo_diff(cv::Scalar(5,5,5)).up_diff(cv::Scalar(10,10,10)), &op);
}

TEST(FloodFillTest, BGRFillsHomogeneousRegion) {
    Image<BGR> img(cv::Mat(50, 50, CV_8UC3, cv::Scalar(0, 0, 0)));
    auto result = FloodFill{}(img, {25, 25}, cv::Scalar(0, 255, 0));
    EXPECT_EQ(result.rows(), 50);
    EXPECT_EQ(result.cols(), 50);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(10, 10)[1], 255);  // green channel
}

TEST(FloodFillTest, GrayFillsHomogeneousRegion) {
    Image<Gray> img(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    auto result = FloodFill{}(img, {25, 25}, static_cast<uchar>(200));
    EXPECT_EQ(result.mat().at<uchar>(10, 10), 200u);
    EXPECT_EQ(result.rows(), 50);
}

TEST(FloodFillTest, OriginalImageUnmodified) {
    cv::Mat m(50, 50, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> img(m.clone());
    FloodFill{}(img, {25, 25}, cv::Scalar(0, 255, 0));
    // img was passed by const ref — original unchanged
    EXPECT_EQ(img.mat().at<cv::Vec3b>(10, 10)[1], 0);
}

TEST(FloodFillTest, SeedOutsideBoundsThrows) {
    Image<BGR> img(cv::Mat(50, 50, CV_8UC3, cv::Scalar(0)));
    EXPECT_THROW(FloodFill{}(img, {-1,  0}, cv::Scalar(0)), improc::ParameterError);
    EXPECT_THROW(FloodFill{}(img, {50,  0}, cv::Scalar(0)), improc::ParameterError);
    EXPECT_THROW(FloodFill{}(img, { 0, -1}, cv::Scalar(0)), improc::ParameterError);
    EXPECT_THROW(FloodFill{}(img, { 0, 50}, cv::Scalar(0)), improc::ParameterError);
}

TEST(FloodFillTest, LoUpDiffToleranceLimitsSpread) {
    // Create a 20x20 image: left half pixel value 100, right half 150
    cv::Mat m(20, 20, CV_8UC1, cv::Scalar(100));
    m.colRange(10, 20).setTo(cv::Scalar(150));
    Image<Gray> img(m);

    // Tight tolerance (5) — fill should NOT cross the boundary
    auto tight = FloodFill{}.lo_diff(cv::Scalar(5)).up_diff(cv::Scalar(5))(
        img, {5, 5}, static_cast<uchar>(200));
    EXPECT_EQ(tight.mat().at<uchar>(5, 5),  200u);  // filled
    EXPECT_EQ(tight.mat().at<uchar>(5, 15), 150u);  // untouched

    // Loose tolerance (60) — fill SHOULD cross the boundary
    auto loose = FloodFill{}.lo_diff(cv::Scalar(60)).up_diff(cv::Scalar(60))(
        img, {5, 5}, static_cast<uchar>(200));
    EXPECT_EQ(loose.mat().at<uchar>(5, 15), 200u);  // also filled
}
