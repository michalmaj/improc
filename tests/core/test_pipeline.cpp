// tests/core/test_pipeline.cpp
#include <gtest/gtest.h>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(PipelineTest, BGRToGrayViaFunctor) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> bgr(mat);
    Image<Gray> gray = bgr | ToGray{};
    EXPECT_EQ(gray.mat().type(), CV_8UC1);
    EXPECT_EQ(gray.rows(), 50);
}

TEST(PipelineTest, GrayToBGRViaFunctor) {
    cv::Mat mat(50, 50, CV_8UC1, cv::Scalar(128));
    Image<Gray> gray(mat);
    Image<BGR> bgr = gray | ToBGR{};
    EXPECT_EQ(bgr.mat().type(), CV_8UC3);
}

TEST(PipelineTest, GrayToFloat32ViaFunctor) {
    cv::Mat mat(1, 1, CV_8UC1, cv::Scalar(255));
    Image<Gray> gray(mat);
    Image<Float32> f = gray | ToFloat32{};
    EXPECT_EQ(f.mat().type(), CV_32FC1);
    EXPECT_NEAR(f.mat().at<float>(0, 0), 1.0f, 1e-5f);
}

TEST(PipelineTest, ChainedBGRToGrayToFloat32) {
    cv::Mat mat(50, 60, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> bgr(mat);
    Image<Float32> result = bgr | ToGray{} | ToFloat32{};
    EXPECT_EQ(result.mat().type(), CV_32FC1);
    EXPECT_EQ(result.rows(), 50);
    EXPECT_EQ(result.cols(), 60);
}
