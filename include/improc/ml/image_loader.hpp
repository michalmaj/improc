//
// Created by Michał Maj on 13/04/2025.
//

#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <filesystem>
#include <unordered_set>
#include <expected>
#include "improc/error.hpp"

namespace improc::ml {

/**
 * @brief Loader for reading all valid images from a given directory.
 *
 * Supported extensions: .jpg, .jpeg, .png
 */
class ImageLoader {
public:
  /// @brief Constructs an ImageLoader with no images loaded.
  ImageLoader() = default;

  /**
   * @brief Load all valid images from the given directory.
   * @param dir_path Path to directory containing image files.
   * @throws FileNotFoundError if the path is not a directory.
   */
  void load_images(const std::filesystem::path& dir_path);

  /**
   * @brief Returns loaded images, or an error message if empty.
   * @return std::expected with vector of cv::Mat images.
   */
  [[nodiscard]]
  std::expected<std::vector<cv::Mat>, improc::Error> get_images();

private:
  std::vector<cv::Mat> images_;
  std::string          last_dir_;
  inline static std::unordered_set<std::string> valid_extensions_{ ".jpg", ".jpeg", ".png" };

  /// @brief Returns a lowercase copy of @p str.
  static std::string to_lower(const std::string& str);
};

} // namespace improc::ml
