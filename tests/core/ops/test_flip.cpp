#include <gtest/gtest.h>
#include "improc/core/ops/flip.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(FlipTest, HorizontalFlipMovesPixel) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    mat.at<cv::Vec3b>(0, 0) = {255, 0, 0};  // top-left
    Image<BGR> img(mat);
    Image<BGR> result = img | Flip{Axis::Horizontal};
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 1), (cv::Vec3b{255, 0, 0}));  // now top-right
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0), (cv::Vec3b{0,   0, 0}));
}

TEST(FlipTest, VerticalFlipMovesPixel) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    mat.at<cv::Vec3b>(0, 0) = {0, 255, 0};  // top-left
    Image<BGR> img(mat);
    Image<BGR> result = img | Flip{Axis::Vertical};
    EXPECT_EQ(result.mat().at<cv::Vec3b>(1, 0), (cv::Vec3b{0, 255, 0}));  // now bottom-left
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0), (cv::Vec3b{0,   0, 0}));  // origin cleared
}

TEST(FlipTest, BothAxesFlipsCompletely) {
    cv::Mat mat(2, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    mat.at<cv::Vec3b>(0, 0) = {0, 0, 255};  // top-left
    Image<BGR> img(mat);
    Image<BGR> result = img | Flip{Axis::Both};
    EXPECT_EQ(result.mat().at<cv::Vec3b>(1, 1), (cv::Vec3b{0, 0, 255}));  // now bottom-right
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0), (cv::Vec3b{0, 0,   0}));  // origin cleared
}

TEST(FlipTest, PreservesDimensionsAndType) {
    cv::Mat mat(50, 80, CV_8UC1);
    Image<Gray> img(mat);
    Image<Gray> result = img | Flip{Axis::Horizontal};
    EXPECT_EQ(result.rows(), 50);
    EXPECT_EQ(result.cols(), 80);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}
