// Created by Michał Maj on 14/04/2025.

#pragma once

#include <opencv2/objdetect.hpp>
#include "improc/ml/model_loader_base.hpp"

namespace improc::ml {

/**
 * @brief Loader class for Haar Cascade XML models using CRTP.
 */
class HaarCascadeLoader : public ModelLoaderBase<HaarCascadeLoader, cv::CascadeClassifier> {
public:
  void load_impl(const std::filesystem::path& path);
  std::expected<cv::CascadeClassifier, std::string> get_impl() const;

private:
  cv::CascadeClassifier classifier_;
};

} // namespace improc::ml
