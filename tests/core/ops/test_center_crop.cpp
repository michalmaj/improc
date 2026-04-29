// tests/core/ops/test_center_crop.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/core/ops/center_crop.hpp"

using namespace improc::core;

TEST(CenterCropTest, ValidDimensionsProducesCorrectSize) {
    cv::Mat mat(100, 200, CV_8UC3);
    Image<BGR> img(mat);
    Image<BGR> result = img | CenterCrop{}.width(50).height(40);
    EXPECT_EQ(result.cols(), 50);
    EXPECT_EQ(result.rows(), 40);
}

TEST(CenterCropTest, CropOriginatesFromCenter) {
    cv::Mat mat(10, 10, CV_8UC1);
    for (int r = 0; r < 10; ++r)
        for (int c = 0; c < 10; ++c)
            mat.at<uchar>(r, c) = static_cast<uchar>(c);
    Image<Gray> img(mat);
    Image<Gray> result = img | CenterCrop{}.width(4).height(10);
    EXPECT_EQ(result.mat().at<uchar>(0, 0), 3);
}

TEST(CenterCropTest, PreservesFormat) {
    cv::Mat mat(100, 100, CV_8UC1);
    Image<Gray> img(mat);
    Image<Gray> result = img | CenterCrop{}.width(50).height(50);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(CenterCropTest, ResultOwnsItsMemory) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(10, 20, 30));
    Image<BGR> img(mat);
    Image<BGR> cropped = img | CenterCrop{}.width(50).height(50);
    img.mat().setTo(cv::Scalar(0, 0, 0));
    EXPECT_EQ(cropped.mat().at<cv::Vec3b>(0, 0), (cv::Vec3b{10, 20, 30}));
}

TEST(CenterCropTest, ExactSizeReturnsFullImage) {
    cv::Mat mat(64, 64, CV_8UC3);
    Image<BGR> img(mat);
    Image<BGR> result = img | CenterCrop{}.width(64).height(64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.rows(), 64);
}

TEST(CenterCropTest, ThrowsOnMissingWidth) {
    cv::Mat mat(100, 100, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | CenterCrop{}.height(50)), improc::ParameterError);
}

TEST(CenterCropTest, ThrowsOnMissingHeight) {
    cv::Mat mat(100, 100, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | CenterCrop{}.width(50)), improc::ParameterError);
}

TEST(CenterCropTest, ThrowsWhenCropWiderThanImage) {
    cv::Mat mat(100, 50, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | CenterCrop{}.width(100).height(50)), improc::ParameterError);
}

TEST(CenterCropTest, ThrowsWhenCropTallerThanImage) {
    cv::Mat mat(50, 100, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | CenterCrop{}.width(50).height(100)), improc::ParameterError);
}

TEST(CenterCropTest, ThrowsOnNonPositiveDimension) {
    EXPECT_THROW(CenterCrop{}.width(0),   improc::ParameterError);
    EXPECT_THROW(CenterCrop{}.height(-1), improc::ParameterError);
}
