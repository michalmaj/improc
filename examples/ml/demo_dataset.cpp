//
// Created by Michał Maj on 13/04/2025.
//

#include "improc/ml/dataset.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>

int main() {
  improc::ml::Dataset dataset;

  auto dataset_path = std::filesystem::path(__FILE__).parent_path() / "demo_data";

  auto result = dataset.load_from_directory(dataset_path, 0.2f, 0.1f, 5);
  if (!result) {
    std::cerr << "Error loading dataset: " << result.error() << std::endl;
    return 1;
  }

  std::cout << "Train: " << dataset.train_images().size() << " samples\n";
  std::cout << "Val:   " << dataset.val_images().size() << " samples\n";
  std::cout << "Test:  " << dataset.test_images().size() << " samples\n";

  std::cout << "\nClass mapping:\n";
  for (const auto& [name, id] : dataset.class_mapping()) {
    std::cout << id << " -> " << name << '\n';
  }

  if (!dataset.train_images().empty()) {
    cv::imshow("First training image", dataset.train_images().front());
    std::cout << "Label: " << dataset.train_labels().front() << '\n';
    cv::waitKey(0);
  }

  return 0;
}
