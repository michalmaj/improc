// Created by Michał Maj on 14/04/2025.

#include "improc/ml/haar_cascade_loader.hpp"
#include "improc/exceptions.hpp"
#include "improc/error.hpp"

namespace improc::ml {

void HaarCascadeLoader::load_impl(const std::filesystem::path& path) {
  if (!classifier_.load(path.string()))
    throw improc::ModelError{path, "CascadeClassifier::load() failed"};
}

std::expected<cv::CascadeClassifier, improc::Error> HaarCascadeLoader::get_impl() const {
  if (classifier_.empty())
    return std::unexpected(improc::Error::invalid_model_file("", "Haar cascade model is empty"));
  return classifier_;
}

} // namespace improc::ml
