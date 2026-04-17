// tests/core/test_image.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/core/image.hpp"

using namespace improc::core;

TEST(ImageTest, ConstructFromValidBGRMat) {
    cv::Mat mat(100, 100, CV_8UC3);
    EXPECT_NO_THROW(Image<BGR> img(mat));
}

TEST(ImageTest, ThrowsOnWrongType) {
    cv::Mat mat(100, 100, CV_8UC1);  // Gray mat, not BGR
    EXPECT_THROW((Image<BGR>{mat}), improc::ParameterError);
}

TEST(ImageTest, RowsColsReflectMat) {
    cv::Mat mat(50, 80, CV_8UC1);
    Image<Gray> img(mat);
    EXPECT_EQ(img.rows(), 50);
    EXPECT_EQ(img.cols(), 80);
}

TEST(ImageTest, EmptyMatThrows) {
    EXPECT_THROW(Image<BGR>(cv::Mat(0, 0, CV_8UC3)), improc::ParameterError);
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

TEST(ImageTest, CloneIsDeepCopy) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));
    Image<BGR> original(mat);
    Image<BGR> copy = original.clone();
    copy.mat().at<cv::Vec3b>(0, 0) = {0, 0, 0};
    EXPECT_EQ(original.mat().at<cv::Vec3b>(0, 0), (cv::Vec3b{10, 20, 30}));
}

TEST(ImageTest, ShallowCopySharesData) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));
    Image<BGR> original(mat);
    Image<BGR> shallow = original;          // copy constructor — shallow
    shallow.mat().at<cv::Vec3b>(0, 0) = {0, 0, 0};
    EXPECT_EQ(original.mat().at<cv::Vec3b>(0, 0), (cv::Vec3b{0, 0, 0}));  // shared data changed
}
