// Created by Michał Maj on 14/04/2025.

#pragma once

#include <filesystem>
#include <unordered_set>
#include <string>
#include <expected>
#include <format>
#include <ranges>

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
    auto expected = check_path(path);
    if (!expected) {
      throw std::runtime_error(expected.error());
    }
    static_cast<Derived*>(this)->load_impl(expected.value());
  }

  [[nodiscard]] std::expected<ModelType, std::string> get_model() const {
    return static_cast<const Derived*>(this)->get_impl();
  }

  static std::expected<std::filesystem::path, std::string> check_path(const std::filesystem::path& path) {
    if (std::filesystem::is_regular_file(path) &&
        valid_extensions().contains(std::ranges::to<std::string>(path.extension().string() | std::views::transform(::tolower)))) {
      return path;
        }
    return std::unexpected(std::format("Invalid model file: {}", path.string()));
  }

protected:
  static const std::unordered_set<std::string>& valid_extensions() {
    static const std::unordered_set<std::string> exts = { ".yml", ".yaml", ".xml" };
    return exts;
  }
};

} // namespace improc::ml
