//
// Created by Michał Maj on 09/04/2025.
//

#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include "improc/io/camera_capture.hpp"

using improc::io::CameraCapture;

TEST(CameraCaptureTest, CanStartAndGrabFrame) {
  CameraCapture camera(0); // Camera ID 0 (default)
  std::this_thread::sleep_for(std::chrono::milliseconds(2500)); // Allow time for initialization

  auto frame = camera.getFrame();
  EXPECT_FALSE(frame.empty()) << "Frame from camera is empty";
}

TEST(CameraCaptureTest, CanStopCapture) {
  CameraCapture camera(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  camera.stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  auto frame = camera.getFrame();
  SUCCEED() << "CameraCapture stopped successfully and getFrame() did not crash.";
}

TEST(CameraCaptureTest, StopIsIdempotent) {
  CameraCapture camera(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  camera.stop();
  camera.stop();
  SUCCEED() << "Multiple calls to stop() work correctly.";
}

TEST(CameraCaptureTest, InvalidCameraIDReturnsEmptyFrame) {
  CameraCapture camera(9999);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  auto frame = camera.getFrame();
  EXPECT_TRUE(frame.empty()) << "Frame should be empty for an invalid camera ID.";
}
