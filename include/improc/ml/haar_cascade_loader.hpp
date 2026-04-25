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
  /**
   * @brief Loads the Haar Cascade XML file into the internal classifier.
   * @param path Validated path to the `.xml` model file.
   * @throws improc::ModelError if `cv::CascadeClassifier::load` fails.
   */
  void load_impl(const std::filesystem::path& path);

  /**
   * @brief Returns the loaded classifier, or an error if not yet loaded.
   * @return The `cv::CascadeClassifier` on success, or an `improc::Error`.
   */
  std::expected<cv::CascadeClassifier, improc::Error> get_impl() const;

private:
  cv::CascadeClassifier classifier_;
};

} // namespace improc::ml
