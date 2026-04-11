// tests/core/ops/test_float32c3.cpp
#include <gtest/gtest.h>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(Float32C3Test, ConvertBGRToFloat32C3PreservesDimensions) {
    cv::Mat mat(50, 60, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> bgr(mat);
    Image<Float32C3> result = convert<Float32C3, BGR>(bgr);
    EXPECT_EQ(result.mat().type(), CV_32FC3);
    EXPECT_EQ(result.rows(), 50);
    EXPECT_EQ(result.cols(), 60);
}

TEST(Float32C3Test, ConvertBGRToFloat32C3NormalizesValues) {
    cv::Mat mat(1, 1, CV_8UC3, cv::Scalar(255, 255, 255));
    Image<BGR> bgr(mat);
    Image<Float32C3> result = convert<Float32C3, BGR>(bgr);
    auto px = result.mat().at<cv::Vec3f>(0, 0);
    EXPECT_NEAR(px[0], 1.0f, 1e-5f);
    EXPECT_NEAR(px[1], 1.0f, 1e-5f);
    EXPECT_NEAR(px[2], 1.0f, 1e-5f);
}

TEST(Float32C3Test, ToFloat32C3FunctorViaPipeline) {
    cv::Mat mat(50, 60, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> bgr(mat);
    Image<Float32C3> result = bgr | ToFloat32C3{};
    EXPECT_EQ(result.mat().type(), CV_32FC3);
}
