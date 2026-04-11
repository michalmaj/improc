#include <gtest/gtest.h>
#include "improc/core/ops/rotate.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(RotateTest, PreservesDimensions) {
    cv::Mat mat(100, 200, CV_8UC3);
    Image<BGR> img(mat);
    Image<BGR> result = img | Rotate{}.angle(45.0);
    EXPECT_EQ(result.rows(), 100);
    EXPECT_EQ(result.cols(), 200);
}

TEST(RotateTest, PreservesType) {
    cv::Mat mat(50, 50, CV_8UC1);
    Image<Gray> img(mat);
    Image<Gray> result = img | Rotate{}.angle(90.0);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(RotateTest, WorksWithScaleOption) {
    cv::Mat mat(100, 100, CV_8UC3);
    Image<BGR> img(mat);
    Image<BGR> result = img | Rotate{}.angle(30.0).scale(0.5);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.rows(), 100);
    EXPECT_EQ(result.cols(), 100);
}

TEST(RotateTest, ThrowsOnNonPositiveScale) {
    EXPECT_THROW(Rotate{}.scale(0.0),  std::invalid_argument);
    EXPECT_THROW(Rotate{}.scale(-1.0), std::invalid_argument);
}

TEST(RotateTest, WorksOnFloat32) {
    cv::Mat mat(50, 50, CV_32FC1);
    Image<Float32> img(mat);
    Image<Float32> result = img | Rotate{}.angle(45.0);
    EXPECT_EQ(result.mat().type(), CV_32FC1);
}

TEST(RotateTest, ThrowsWithoutAngle) {
    cv::Mat mat(50, 50, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | Rotate{}), std::invalid_argument);
}
