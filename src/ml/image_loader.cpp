//
// Created by Michał Maj on 13/04/2025.
//

#include "improc/ml/image_loader.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <format>
#include <ranges>
#include <algorithm>

namespace improc::ml {

void ImageLoader::load_images(const std::filesystem::path& dir_path) {
  if (!std::filesystem::is_directory(dir_path)) {
    std::cerr << std::format("Wrong dir_path: {}\n", dir_path.string());
    throw std::runtime_error("Can't load data!\n");
  }

  images_.clear();

  for (const auto& file : std::filesystem::directory_iterator(dir_path)) {
    if (!std::filesystem::is_regular_file(file)) continue;

    const auto ext = to_lower(file.path().extension().string());

    if (valid_extensions_.contains(ext)) {
      cv::Mat img = cv::imread(file.path().string());
      if (img.empty()) {
        std::cerr << std::format("Can't load file: {}\n", file.path().string());
        continue;
      }
      images_.emplace_back(std::move(img));
    }
  }
}

std::expected<std::vector<cv::Mat>, std::string> ImageLoader::get_images() {
  if (images_.empty()) {
    return std::unexpected("There is no images to process!\n");
  }

  return std::move(images_);
}

std::string ImageLoader::to_lower(const std::string& str) {
  std::string result;
  result.reserve(str.size());
  std::ranges::transform(str, std::back_inserter(result), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
  });
  return result;
}

} // namespace improc::ml
