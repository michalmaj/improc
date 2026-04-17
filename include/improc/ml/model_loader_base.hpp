// Created by Michał Maj on 14/04/2025.

#pragma once

#include <filesystem>
#include <unordered_set>
#include <string>
#include <expected>
#include <format>
#include <ranges>
#include "improc/exceptions.hpp"
#include "improc/error.hpp"

namespace improc::ml {

/**
 * @brief Base class for CRTP-based model loaders.
 *        Provides common path validation and model loading logic.
 *
 * @tparam Derived Concrete loader implementing load_impl and get_impl
 * @tparam ModelType Return type of the model (e.g., cv::Ptr<cv::ml::SVM>, cv::CascadeClassifier)
 */
template<typename Derived, typename ModelType>
class ModelLoaderBase {
public:
  void load_model(const std::filesystem::path& path) {
    auto result = check_path(path);
    if (!result)
      throw improc::ModelError{path, result.error().message};
    static_cast<Derived*>(this)->load_impl(result.value());
  }

  [[nodiscard]] std::expected<ModelType, improc::Error> get_model() const {
    return static_cast<const Derived*>(this)->get_impl();
  }

  static std::expected<std::filesystem::path, improc::Error> check_path(const std::filesystem::path& path) {
    auto ext = std::ranges::to<std::string>(
        path.extension().string() | std::views::transform(::tolower));
    if (std::filesystem::is_regular_file(path) && valid_extensions().contains(ext))
      return path;
    if (!std::filesystem::exists(path))
      return std::unexpected(improc::Error::invalid_model_file(
          path.string(), "file not found"));
    return std::unexpected(improc::Error::invalid_model_file(
        path.string(),
        std::format("unsupported extension '{}', expected .yml/.yaml/.xml", ext)));
  }

protected:
  static const std::unordered_set<std::string>& valid_extensions() {
    static const std::unordered_set<std::string> exts = { ".yml", ".yaml", ".xml" };
    return exts;
  }
};

} // namespace improc::ml
