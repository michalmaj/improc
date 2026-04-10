// tests/core/test_image.cpp
#include <gtest/gtest.h>
#include "improc/core/image.hpp"

using namespace improc::core;

TEST(ImageTest, ConstructFromValidBGRMat) {
    cv::Mat mat(100, 100, CV_8UC3);
    EXPECT_NO_THROW(Image<BGR> img(mat));
}

TEST(ImageTest, ThrowsOnWrongType) {
    cv::Mat mat(100, 100, CV_8UC1);  // Gray mat, not BGR
    EXPECT_THROW((Image<BGR>{mat}), std::invalid_argument);
}

TEST(ImageTest, RowsColsReflectMat) {
    cv::Mat mat(50, 80, CV_8UC1);
    Image<Gray> img(mat);
    EXPECT_EQ(img.rows(), 50);
    EXPECT_EQ(img.cols(), 80);
}

TEST(ImageTest, EmptyDetected) {
    Image<BGR> img(cv::Mat(0, 0, CV_8UC3));
    EXPECT_TRUE(img.empty());
}

TEST(ImageTest, NonEmptyDetected) {
    Image<BGR> img(cv::Mat(10, 10, CV_8UC3));
    EXPECT_FALSE(img.empty());
}

TEST(ImageTest, MatAccessorReturnsSameMat) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(1, 2, 3));
    Image<BGR> img(mat);
    EXPECT_EQ(img.mat().data, mat.data);  // same underlying data
}
