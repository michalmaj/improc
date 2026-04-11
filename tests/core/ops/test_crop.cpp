// tests/core/ops/test_crop.cpp
#include <gtest/gtest.h>
#include "improc/core/pipeline.hpp"
#include "improc/core/ops/crop.hpp"

using namespace improc::core;

TEST(CropTest, ValidROIProducesCorrectDimensions) {
    cv::Mat mat(100, 200, CV_8UC3);
    Image<BGR> img(mat);
    Image<BGR> result = img | Crop{}.x(10).y(10).width(50).height(40);
    EXPECT_EQ(result.cols(), 50);
    EXPECT_EQ(result.rows(), 40);
}

TEST(CropTest, PreservesFormat) {
    cv::Mat mat(100, 200, CV_8UC1);
    Image<Gray> img(mat);
    Image<Gray> result = img | Crop{}.x(0).y(0).width(50).height(50);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(CropTest, ResultOwnsItsMemory) {
    cv::Mat mat(100, 200, CV_8UC3, cv::Scalar(10, 20, 30));
    Image<BGR> img(mat);
    Image<BGR> cropped = img | Crop{}.x(0).y(0).width(50).height(50);
    img.mat().setTo(cv::Scalar(0, 0, 0));
    EXPECT_EQ(cropped.mat().at<cv::Vec3b>(0, 0), (cv::Vec3b{10, 20, 30}));
}

TEST(CropTest, ThrowsOnMissingWidth) {
    cv::Mat mat(100, 200, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | Crop{}.x(0).y(0).height(50)), std::invalid_argument);
}

TEST(CropTest, ThrowsOnMissingX) {
    cv::Mat mat(100, 200, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | Crop{}.y(0).width(50).height(50)), std::invalid_argument);
}

TEST(CropTest, ThrowsOnROIOutOfBounds) {
    cv::Mat mat(100, 200, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | Crop{}.x(0).y(0).width(300).height(50)), std::invalid_argument);
}

TEST(CropTest, ThrowsOnROIExceedingHeight) {
    cv::Mat mat(100, 200, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | Crop{}.x(0).y(50).width(50).height(100)), std::invalid_argument);
}
