// tests/core/ops/test_resize.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/core/ops/resize.hpp"

using namespace improc::core;

TEST(ResizeTest, BothDimensionsExplicit) {
    cv::Mat mat(100, 200, CV_8UC3);
    Image<BGR> img(mat);
    Image<BGR> result = img | Resize{}.width(50).height(50);
    EXPECT_EQ(result.cols(), 50);
    EXPECT_EQ(result.rows(), 50);
}

TEST(ResizeTest, WidthOnlyPreservesAspectRatio) {
    cv::Mat mat(100, 200, CV_8UC3);  // cols=200, rows=100 → ratio 2:1
    Image<BGR> img(mat);
    Image<BGR> result = img | Resize{}.width(100);
    EXPECT_EQ(result.cols(), 100);
    EXPECT_EQ(result.rows(), 50);
}

TEST(ResizeTest, HeightOnlyPreservesAspectRatio) {
    cv::Mat mat(100, 200, CV_8UC3);  // cols=200, rows=100 → ratio 2:1
    Image<BGR> img(mat);
    Image<BGR> result = img | Resize{}.height(50);
    EXPECT_EQ(result.rows(), 50);
    EXPECT_EQ(result.cols(), 100);
}

TEST(ResizeTest, WorksOnGrayFormat) {
    cv::Mat mat(100, 200, CV_8UC1);
    Image<Gray> img(mat);
    Image<Gray> result = Resize{}.width(50).height(50)(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.cols(), 50);
}

TEST(ResizeTest, WorksOnFloat32Format) {
    cv::Mat mat(100, 200, CV_32FC1);
    Image<Float32> img(mat);
    Image<Float32> result = img | Resize{}.width(40).height(40);
    EXPECT_EQ(result.mat().type(), CV_32FC1);
    EXPECT_EQ(result.cols(), 40);
    EXPECT_EQ(result.rows(), 40);
}

TEST(ResizeTest, ThrowsWithNoDimensions) {
    cv::Mat mat(100, 200, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | Resize{}), improc::ParameterError);
}

TEST(ResizeTest, ThrowsOnZeroWidth) {
    EXPECT_THROW(Resize{}.width(0), improc::ParameterError);
}

TEST(ResizeTest, ThrowsOnNegativeHeight) {
    EXPECT_THROW(Resize{}.height(-10), improc::ParameterError);
}
