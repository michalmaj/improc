//
// Created by Michał Maj on 13/04/2025.
//

#include "improc/ml/image_loader.hpp"
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <format>
#include <ranges>
#include <algorithm>
#include "improc/exceptions.hpp"

namespace improc::ml {

void ImageLoader::load_images(const std::filesystem::path& dir_path) {
  if (!std::filesystem::is_directory(dir_path))
    throw improc::FileNotFoundError{dir_path};

  images_.clear();
  last_dir_ = dir_path.string();

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

std::expected<std::vector<cv::Mat>, improc::Error> ImageLoader::get_images() {
  if (images_.empty())
    return std::unexpected(improc::Error::no_images(last_dir_));
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
