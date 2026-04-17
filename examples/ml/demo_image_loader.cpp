//
// Created by Michał Maj on 13/04/2025.
//

#include "improc/ml/image_loader.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>

using improc::ml::ImageLoader;

int main() {
  ImageLoader loader;

  try {
    auto demo_dir = std::filesystem::path(__FILE__).parent_path() / "demo_images";
    loader.load_images(demo_dir);
    auto result = loader.get_images();

    if (result) {
      std::cout << "Loaded " << result->size() << " images.\n";
      cv::imshow("First Image", result->front());
      cv::waitKey(0);
    } else {
      std::cerr << result.error().message << std::endl;
    }

  } catch (const std::exception& e) {
    std::cerr << "Error during loading: " << e.what() << std::endl;
  }

  return 0;
}
