// tests/io/test_image_io.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "improc/io/image_io.hpp"
#include "improc/error.hpp"

using namespace improc::core;
using improc::io::imread;
using improc::io::imwrite;
using improc::Error;

static constexpr const char* kTestPng    = "/tmp/test_improc_img.png";
static constexpr const char* kRoundTrip  = "/tmp/test_improc_rt.png";
static constexpr const char* kGrayPng    = "/tmp/test_improc_gray.png";

class ImageIOTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Write a 4x4 BGR PNG to /tmp for read tests
        cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(100, 150, 200));
        cv::imwrite(kTestPng, mat);
    }
};

// 1. Read a BGR PNG as BGR
TEST_F(ImageIOTest, ReadBGRFromPNG) {
    auto result = imread<BGR>(kTestPng);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result->mat().type(), CV_8UC3);
}

// 2. Read the same PNG as Gray
TEST_F(ImageIOTest, ReadGrayFromPNG) {
    auto result = imread<Gray>(kTestPng);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result->mat().type(), CV_8UC1);
}

// 3. Read as Float32C3 — values must be in [0, 1]
TEST_F(ImageIOTest, ReadFloat32C3FromPNG) {
    auto result = imread<Float32C3>(kTestPng);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result->mat().type(), CV_32FC3);

    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(result->mat(), &minVal, &maxVal);
    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 1.0);
}

// 4. Non-existent file returns ImageReadFailed error
TEST_F(ImageIOTest, ReadNonExistentReturnsError) {
    auto result = imread<BGR>("/nonexistent/path/img.png");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, Error::Code::ImageReadFailed);
}

// 5. Write and read back (PNG is lossless — pixel values must be preserved)
TEST_F(ImageIOTest, WriteAndReadRoundTrip) {
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(42, 84, 126));
    Image<BGR> img(mat);

    auto ok = imwrite(kRoundTrip, img);
    ASSERT_TRUE(ok.has_value()) << ok.error().message;

    auto back = imread<BGR>(kRoundTrip);
    ASSERT_TRUE(back.has_value()) << back.error().message;

    // Check the first pixel
    cv::Vec3b pixel = back->mat().at<cv::Vec3b>(0, 0);
    EXPECT_EQ(pixel[0], 42);
    EXPECT_EQ(pixel[1], 84);
    EXPECT_EQ(pixel[2], 126);
}

// 6. Writing to a bad path returns ImageWriteFailed error
TEST_F(ImageIOTest, WriteReturnsErrorOnBadPath) {
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> img(mat);

    auto ok = imwrite("/nonexistent_dir/out.png", img);
    ASSERT_FALSE(ok.has_value());
    EXPECT_EQ(ok.error().code, Error::Code::ImageWriteFailed);
}

// 7. Write and read back a Gray image
TEST_F(ImageIOTest, WriteAndReadGray) {
    cv::Mat mat(4, 4, CV_8UC1, cv::Scalar(77));
    Image<Gray> img(mat);

    auto ok = imwrite(kGrayPng, img);
    ASSERT_TRUE(ok.has_value()) << ok.error().message;

    auto back = imread<Gray>(kGrayPng);
    ASSERT_TRUE(back.has_value()) << back.error().message;
    EXPECT_EQ(back->mat().type(), CV_8UC1);

    uchar pixel = back->mat().at<uchar>(0, 0);
    EXPECT_EQ(pixel, 77);
}
