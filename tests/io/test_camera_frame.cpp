// tests/io/test_camera_frame.cpp
#include <gtest/gtest.h>
#include <chrono>
#include "improc/io/camera_frame.hpp"
#include "improc/error.hpp"

using namespace improc::io;
using namespace improc::core;
using improc::Error;

TEST(CameraFrameTest, DefaultConstruction) {
    CameraFrame f;
    EXPECT_FALSE(f.rgb.has_value());
    EXPECT_FALSE(f.depth.has_value());
    EXPECT_TRUE(f.source_id.empty());
}

TEST(CameraFrameTest, RgbAndTimestampRoundTrip) {
    cv::Mat mat = cv::Mat::zeros(4, 4, CV_8UC3);
    CameraFrame f;
    f.rgb = Image<BGR>(mat);
    f.timestamp = std::chrono::steady_clock::now();
    f.source_id = "test:0";
    ASSERT_TRUE(f.rgb.has_value());
    EXPECT_EQ(f.rgb->mat().rows, 4);
    EXPECT_FALSE(f.depth.has_value());
    EXPECT_EQ(f.source_id, "test:0");
}

TEST(CameraFrameTest, DepthOptionalPopulated) {
    CameraFrame f;
    cv::Mat depth_mat(4, 4, CV_32FC1, cv::Scalar(1.5f));
    f.depth = Image<Float32>(depth_mat);
    ASSERT_TRUE(f.depth.has_value());
    EXPECT_FLOAT_EQ(f.depth->mat().at<float>(0, 0), 1.5f);
}

TEST(CameraFrameErrorTest, TimeoutCode) {
    auto err = Error::timeout("oak-d");
    EXPECT_EQ(err.code, Error::Code::Timeout);
    EXPECT_FALSE(err.message.empty());
}
