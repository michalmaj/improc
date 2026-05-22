// tests/calib/test_undistort.cpp
#include <gtest/gtest.h>
#include "improc/calib/pipeline.hpp"

using namespace improc::calib;
using namespace improc::core;

namespace {
cv::Mat make_K(int w, int h) {
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0,0) = static_cast<double>(w);      // fx
    K.at<double>(1,1) = static_cast<double>(h);      // fy
    K.at<double>(0,2) = w / 2.0;                     // cx
    K.at<double>(1,2) = h / 2.0;                     // cy
    return K;
}
cv::Mat zero_dist() { return cv::Mat::zeros(1, 5, CV_64F); }
} // namespace

TEST(UndistortTest, OutputSizeMatchesInput_BGR) {
    Image<BGR> img(cv::Mat(120, 160, CV_8UC3, cv::Scalar(50,100,150)));
    auto result = img | Undistort{}.K(make_K(160,120)).dist(zero_dist());
    EXPECT_EQ(result.mat().rows, 120);
    EXPECT_EQ(result.mat().cols, 160);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(UndistortTest, OutputSizeMatchesInput_Gray) {
    Image<Gray> img(cv::Mat(120, 160, CV_8UC1, cv::Scalar(128)));
    auto result = img | Undistort{}.K(make_K(160,120)).dist(zero_dist());
    EXPECT_EQ(result.mat().rows, 120);
    EXPECT_EQ(result.mat().cols, 160);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(UndistortTest, ZeroDistortionPreservesPixels) {
    cv::Mat raw(120, 160, CV_8UC3, cv::Scalar(100, 150, 200));
    auto result = Image<BGR>(raw) | Undistort{}.K(make_K(160,120)).dist(zero_dist());
    cv::Mat diff;
    cv::absdiff(result.mat(), raw, diff);
    EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(UndistortTest, ThrowsWhenKNotSet) {
    Image<BGR> img(cv::Mat(120, 160, CV_8UC3, cv::Scalar(0)));
    EXPECT_THROW(img | Undistort{}.dist(zero_dist()), std::invalid_argument);
}

TEST(UndistortTest, ThrowsWhenDistNotSet) {
    Image<BGR> img(cv::Mat(120, 160, CV_8UC3, cv::Scalar(0)));
    EXPECT_THROW(img | Undistort{}.K(make_K(160,120)), std::invalid_argument);
}

TEST(UndistortTest, FluentSettersReturnThis) {
    Undistort op;
    EXPECT_EQ(&op.K(make_K(160,120)), &op);
    EXPECT_EQ(&op.dist(zero_dist()), &op);
}

TEST(UndistortMapTest, MapsHaveCorrectSize) {
    cv::Size img_size{160, 120};
    auto maps = UndistortMap{}.K(make_K(160,120)).dist(zero_dist())(img_size);
    EXPECT_EQ(maps.map1.rows, 120);
    EXPECT_EQ(maps.map1.cols, 160);
    EXPECT_EQ(maps.map2.rows, 120);
    EXPECT_EQ(maps.map2.cols, 160);
}

TEST(UndistortMapTest, MapsAreUsableWithCoreRemap) {
    Image<BGR> img(cv::Mat(120, 160, CV_8UC3, cv::Scalar(50,100,150)));
    auto maps = UndistortMap{}.K(make_K(160,120)).dist(zero_dist())({160, 120});
    auto result = img | improc::core::Remap{maps.map1, maps.map2};
    EXPECT_EQ(result.mat().rows, 120);
    EXPECT_EQ(result.mat().cols, 160);
}

TEST(UndistortMapTest, ThrowsOnZeroImageSize) {
    EXPECT_THROW(UndistortMap{}.K(make_K(160,120)).dist(zero_dist())({0, 0}),
                 std::invalid_argument);
}

TEST(UndistortMapTest, ThrowsWhenKNotSet) {
    EXPECT_THROW(UndistortMap{}.dist(zero_dist())({160, 120}), std::invalid_argument);
}

TEST(UndistortMapTest, ThrowsWhenDistNotSet) {
    EXPECT_THROW(UndistortMap{}.K(make_K(160,120))({160, 120}), std::invalid_argument);
}

TEST(UndistortMapTest, FluentSettersReturnThis) {
    UndistortMap op;
    EXPECT_EQ(&op.K(make_K(160,120)), &op);
    EXPECT_EQ(&op.dist(zero_dist()), &op);
}
