// Created by Michał Maj on 14/04/2025.

#include <opencv2/opencv.hpp>
#include "improc/ml/haar_cascade_loader.hpp"
#include <iostream>

using improc::ml::HaarCascadeLoader;

int main() {
  HaarCascadeLoader loader;
  auto model_path = std::filesystem::path(__FILE__).parent_path() / "models" / "haarcascade_frontalface_default.xml";

  try {
    loader.load_model(model_path);
    auto model = loader.get_model();
    if (!model) {
      std::cerr << "Model load failed: " << model.error() << std::endl;
      return 1;
    }

    cv::VideoCapture cap(1);
    if (!cap.isOpened()) {
      std::cerr << "Cannot open webcam" << std::endl;
      return 1;
    }

    cv::Mat frame;
    while (true) {
      cap >> frame;
      if (frame.empty()) break;

      std::vector<cv::Rect> faces;
      model->detectMultiScale(frame, faces);
      for (const auto& face : faces) {
        cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
      }

      cv::imshow("Face Detection", frame);
      if (cv::waitKey(30) == 27) break; // ESC
    }
  }
  catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
