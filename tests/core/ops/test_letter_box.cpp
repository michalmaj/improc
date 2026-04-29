// tests/core/ops/test_letter_box.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/core/ops/letter_box.hpp"

using namespace improc::core;

TEST(LetterBoxTest, OutputIsExactTargetSize) {
    cv::Mat mat(480, 640, CV_8UC3);
    Image<BGR> img(mat);
    Image<BGR> result = img | LetterBox{}.width(224).height(224);
    EXPECT_EQ(result.cols(), 224);
    EXPECT_EQ(result.rows(), 224);
}

TEST(LetterBoxTest, WideImageFitsToHeight) {
    // 200x100 → 64x64: scale = min(64/200, 64/100) = 0.32
    // resized: 64x32, pad_h = 32 → top=16, bottom=16
    cv::Mat mat(100, 200, CV_8UC3, cv::Scalar(0, 0, 200));
    Image<BGR> img(mat);
    Image<BGR> result = img | LetterBox{}.width(64).height(64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.rows(), 64);
    cv::Vec3b top_pixel = result.mat().at<cv::Vec3b>(0, 32);
    EXPECT_EQ(top_pixel[2], 114);  // default fill: R=114
}

TEST(LetterBoxTest, TallImageFitsToWidth) {
    // 100x200 → 64x64: scale = min(64/100, 64/200) = 0.32
    // resized: 32x64, pad_w = 32 → left=16, right=16
    cv::Mat mat(200, 100, CV_8UC3, cv::Scalar(0, 200, 0));
    Image<BGR> img(mat);
    Image<BGR> result = img | LetterBox{}.width(64).height(64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.rows(), 64);
    cv::Vec3b left_pixel = result.mat().at<cv::Vec3b>(32, 0);
    EXPECT_EQ(left_pixel[1], 114);  // default fill: G=114
}

TEST(LetterBoxTest, SquareImageScalesWithoutPadding) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(50, 50, 50));
    Image<BGR> img(mat);
    Image<BGR> result = img | LetterBox{}.width(64).height(64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.rows(), 64);
}

TEST(LetterBoxTest, SmallImageScalesUp) {
    cv::Mat mat(32, 32, CV_8UC3);
    Image<BGR> img(mat);
    Image<BGR> result = img | LetterBox{}.width(64).height(64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.rows(), 64);
}

TEST(LetterBoxTest, DefaultFillColorIs114) {
    cv::Mat mat(100, 200, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> img(mat);
    Image<BGR> result = img | LetterBox{}.width(64).height(64);
    cv::Vec3b fill = result.mat().at<cv::Vec3b>(0, 32);
    EXPECT_EQ(fill[0], 114);
    EXPECT_EQ(fill[1], 114);
    EXPECT_EQ(fill[2], 114);
}

TEST(LetterBoxTest, CustomFillColorApplied) {
    cv::Mat mat(100, 200, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> img(mat);
    Image<BGR> result = img | LetterBox{}.width(64).height(64).value(cv::Scalar(0, 0, 255));
    cv::Vec3b fill = result.mat().at<cv::Vec3b>(0, 32);
    EXPECT_EQ(fill[0], 0);
    EXPECT_EQ(fill[1], 0);
    EXPECT_EQ(fill[2], 255);
}

TEST(LetterBoxTest, PreservesGrayFormat) {
    cv::Mat mat(100, 200, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(mat);
    Image<Gray> result = img | LetterBox{}.width(64).height(64);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.rows(), 64);
}

TEST(LetterBoxTest, ThrowsOnMissingWidth) {
    cv::Mat mat(100, 100, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | LetterBox{}.height(64)), improc::ParameterError);
}

TEST(LetterBoxTest, ThrowsOnMissingHeight) {
    cv::Mat mat(100, 100, CV_8UC3);
    Image<BGR> img(mat);
    EXPECT_THROW((img | LetterBox{}.width(64)), improc::ParameterError);
}

TEST(LetterBoxTest, ThrowsOnNonPositiveDimension) {
    EXPECT_THROW(LetterBox{}.width(0),   improc::ParameterError);
    EXPECT_THROW(LetterBox{}.height(-1), improc::ParameterError);
}

TEST(LetterBoxTest, PipelineSyntaxWorks) {
    cv::Mat mat(480, 640, CV_8UC3);
    Image<BGR> img(mat);
    Image<BGR> result = img | LetterBox{}.width(416).height(416);
    EXPECT_EQ(result.cols(), 416);
    EXPECT_EQ(result.rows(), 416);
}
