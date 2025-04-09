//
// Created by Michał Maj on 09/04/2025.
//

#include <chrono>
#include <thread>
#include <gtest/gtest.h>
#include "improc/io/camera_capture.hpp"

using namespace improc::io;

// Test case for normal functioning: when camera 0 is available, the frame should not be empty.
TEST(CameraCaptureTest, CanStartAndGrabFrame) {
  CameraCapture camera(0); // Camera ID 0 (default)
  std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // Allow time for the camera to initialize

  auto frame = camera.getFrame();

  // We expect that a valid frame is returned
  EXPECT_FALSE(frame.empty()) << "Frame from camera is empty";
}

// Test case to check if stopping the camera capture terminates the capture thread without error.
TEST(CameraCaptureTest, CanStopCapture) {
  CameraCapture camera(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // Allow time for initialization
  camera.stop();  // Stop camera capture
  std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Give a little time after stopping

  // Even after stopping, getFrame() should work (though it may return an empty frame)
  auto frame = camera.getFrame();
  SUCCEED() << "CameraCapture stopped successfully and getFrame() did not crash.";
}

// Test that calling stop() multiple times does not cause issues (idempotent stop).
TEST(CameraCaptureTest, StopIsIdempotent) {
  CameraCapture camera(0);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // Allow time for initialization
  camera.stop();
  // Call stop() a second time – it should not cause a crash or hang.
  camera.stop();
  SUCCEED() << "Multiple calls to stop() work correctly.";
}

// Test handling of an invalid camera ID. The expectation is that getFrame() returns an empty frame.
TEST(CameraCaptureTest, InvalidCameraIDReturnsEmptyFrame) {
  // Use a camera ID that is unlikely to be valid.
  CameraCapture camera(9999);
  std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // Allow time for initialization attempt

  auto frame = camera.getFrame();
  EXPECT_TRUE(frame.empty()) << "Frame should be empty for an invalid camera ID.";
}
