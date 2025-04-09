//
// Created by Michał Maj on 09/04/2025.
//

#include "improc/io/camera_capture.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <thread>

using improc::io::CameraCapture;

int main() {
  CameraCapture camera(0); // Default camera

  std::cout << "Starting camera. Press ESC to exit." << std::endl;

  while (true) {
    auto frame = camera.getFrame();

    if (!frame.empty()) {
      cv::imshow(camera.getWindowName(), frame);
    }

    // Wait 30 ms for key press, break loop if ESC is pressed
    int key = cv::waitKey(30);
    if (key == 27) { // ESC
      break;
    }
  }

  camera.stop();
  std::cout << "Camera stopped." << std::endl;
  return 0;
}
