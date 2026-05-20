// tests/core/ops/test_moments.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/moments.hpp"
#include "improc/core/image.hpp"

using namespace improc::core;

TEST(MomentsTest, DefaultConstruction) {
    Moments m;
    EXPECT_FALSE(m.binary);
}

TEST(MomentsTest, SetBinaryTrue) {
    Moments m;
    m.binary = true;
    EXPECT_TRUE(m.binary);
}

TEST(MomentsTest, SolidWhiteSquareGray) {
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(0));
    mat(cv::Rect(5, 5, 10, 10)).setTo(255);
    Image<Gray> img(mat);

    Moments m;
    cv::Moments result = m(img);

    double expected = 255.0 * 100.0;
    EXPECT_NEAR(result.m00, expected, 0.5);
}

TEST(MomentsTest, SolidWhiteSquareBinary) {
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(0));
    mat(cv::Rect(5, 5, 10, 10)).setTo(255);
    Image<Gray> img(mat);

    Moments m;
    m.binary = true;
    cv::Moments result = m(img);

    double expected = 100.0;
    EXPECT_NEAR(result.m00, expected, 0.5);
}

TEST(MomentsTest, EmptyImage) {
    cv::Mat mat(20, 20, CV_8UC1, cv::Scalar(0));
    Image<Gray> img(mat);

    Moments m;
    cv::Moments result = m(img);

    EXPECT_NEAR(result.m00, 0.0, 0.5);
}
