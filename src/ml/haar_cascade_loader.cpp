// Created by Michał Maj on 14/04/2025.

#include "improc/ml/haar_cascade_loader.hpp"
#include <stdexcept>

namespace improc::ml {

void HaarCascadeLoader::load_impl(const std::filesystem::path& path) {
  if (!classifier_.load(path.string())) {
    throw std::runtime_error("Failed to load Haar Cascade from: " + path.string());
  }
}

std::expected<cv::CascadeClassifier, std::string> HaarCascadeLoader::get_impl() const {
  if (classifier_.empty()) {
    return std::unexpected("Haar cascade model is empty");
  }
  return classifier_;
}

} // namespace improc::ml
