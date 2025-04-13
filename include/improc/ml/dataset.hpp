// Created by Michał Maj on 13/04/2025.

#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <filesystem>
#include <unordered_map>
#include <expected>
#include <optional>

namespace improc::ml {

/**
 * @brief Class responsible for loading image dataset from class-labeled subdirectories
 *        and splitting it into training, validation, and test sets.
 */
class Dataset {
public:
    Dataset() = default;

    /**
     * @brief Load dataset from directory, split into train/val/test.
     *
     * Expected structure:
     *   root_dir/
     *     class0/
     *       image1.png
     *     class1/
     *       image2.jpg
     *
     * @param root Path to dataset directory with subfolders as classes
     * @param test_ratio Proportion of dataset to be used as test set
     * @param val_ratio Proportion of dataset to be used as validation set
     * @param max_per_class Optional: maximum number of images to load per class
     * @return std::expected with error if loading fails
     */
    std::expected<void, std::string> load_from_directory(const std::filesystem::path& root,
                                                         float test_ratio = 0.2f,
                                                         float val_ratio = 0.1f,
                                                         std::optional<size_t> max_per_class = std::nullopt);

    const std::vector<cv::Mat>& train_images() const;
    const std::vector<cv::Mat>& val_images() const;
    const std::vector<cv::Mat>& test_images() const;

    const std::vector<int>& train_labels() const;
    const std::vector<int>& val_labels() const;
    const std::vector<int>& test_labels() const;

    const std::unordered_map<std::string, int>& class_mapping() const;

    /**
     * @brief Get class name for given integer label
     */
    std::string class_name_for(int label) const;

    /**
     * @brief Set random seed for deterministic splitting
     */
    void set_shuffle_seed(unsigned int seed);

private:
    std::vector<cv::Mat> train_;
    std::vector<cv::Mat> val_;
    std::vector<cv::Mat> test_;

    std::vector<int> train_labels_;
    std::vector<int> val_labels_;
    std::vector<int> test_labels_;

    std::unordered_map<std::string, int> class_to_label_;
    std::unordered_map<int, std::string> label_to_class_;

    std::optional<unsigned int> shuffle_seed_ = std::nullopt;

    void shuffle_and_split(std::vector<cv::Mat>& images, int label, float test_ratio, float val_ratio);
};

} // namespace improc::ml
