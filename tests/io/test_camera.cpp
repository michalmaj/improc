// tests/io/test_camera.cpp
#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include <opencv2/videoio.hpp>
#include "improc/io/webcam_capture.hpp"

using improc::io::WebcamCapture;

static bool camera_available() {
    cv::VideoCapture cap(0);
    return cap.isOpened();
}

TEST(WebcamCaptureTest, CanStartAndGrabFrame) {
    if (!camera_available()) GTEST_SKIP() << "No camera available on this system.";
    WebcamCapture camera(0);
    camera.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    auto frame = camera.getFrame();
    ASSERT_TRUE(frame.has_value()) << frame.error().message;
    EXPECT_TRUE(frame->rgb.has_value());
    EXPECT_FALSE(frame->rgb->mat().empty());
    EXPECT_EQ(frame->source_id, "webcam:0");
    EXPECT_FALSE(frame->depth.has_value());
}

TEST(WebcamCaptureTest, CanStopCapture) {
    if (!camera_available()) GTEST_SKIP() << "No camera available.";
    WebcamCapture camera(0);
    camera.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    camera.stop();
    SUCCEED();
}

TEST(WebcamCaptureTest, StopIsIdempotent) {
    if (!camera_available()) GTEST_SKIP() << "No camera available.";
    WebcamCapture camera(0);
    camera.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    camera.stop();
    camera.stop();
    SUCCEED();
}

TEST(WebcamCaptureTest, InvalidCameraIDReturnsError) {
    WebcamCapture camera(9999);
    camera.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    auto frame = camera.getFrame();
    EXPECT_FALSE(frame.has_value());
    EXPECT_EQ(frame.error().code, improc::Error::Code::CameraUnavailable);
}
