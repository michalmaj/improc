// Created by Michał Maj on 13/04/2025.

#include "improc/ml/dataset.hpp"
#include "improc/ml/image_loader.hpp"

#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <random>
#include <algorithm>
#include <format>
#include <ranges>

namespace improc::ml {

std::expected<void, improc::Error> Dataset::load_from_directory(const std::filesystem::path& root,
                                                               float test_ratio,
                                                               float val_ratio,
                                                               std::optional<size_t> max_per_class) {
    if (!std::filesystem::exists(root) || !std::filesystem::is_directory(root))
        return std::unexpected(improc::Error::directory_not_found(root.string()));

    class_to_label_.clear();
    label_to_class_.clear();
    train_.clear(); val_.clear(); test_.clear();
    train_labels_.clear(); val_labels_.clear(); test_labels_.clear();

    int label_counter = 0;
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!entry.is_directory()) continue;

        const auto& class_dir = entry.path();
        const std::string class_name = class_dir.filename().string();

        ImageLoader loader;
        loader.load_images(class_dir);
        auto result = loader.get_images();

        if (!result)
            return std::unexpected(improc::Error::no_images(
                class_dir.string() + " (class '" + class_name + "'): " + result.error().message));

        auto images = std::move(result.value());
        if (max_per_class && images.size() > *max_per_class) {
            images.resize(*max_per_class);
        }

        class_to_label_[class_name] = label_counter;
        label_to_class_[label_counter] = class_name;

        shuffle_and_split(images, label_counter, test_ratio, val_ratio);

        label_counter++;
    }

    return {};
}

void Dataset::shuffle_and_split(std::vector<cv::Mat>& images, int label, float test_ratio, float val_ratio) {
    if (shuffle_seed_) {
        std::mt19937 rng(*shuffle_seed_);
        std::ranges::shuffle(images, rng);
    } else {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::ranges::shuffle(images, rng);
    }

    size_t total = images.size();
    size_t val_count = static_cast<size_t>(total * val_ratio);
    size_t test_count = static_cast<size_t>(total * test_ratio);
    size_t train_count = total - val_count - test_count;

    auto it = images.begin();
    auto val_end = it + val_count;
    auto test_end = val_end + test_count;

    val_.insert(val_.end(), it, val_end);
    val_labels_.insert(val_labels_.end(), val_count, label);

    test_.insert(test_.end(), val_end, test_end);
    test_labels_.insert(test_labels_.end(), test_count, label);

    train_.insert(train_.end(), test_end, images.end());
    train_labels_.insert(train_labels_.end(), train_count, label);
}

const std::vector<cv::Mat>& Dataset::train_images() const { return train_; }
const std::vector<cv::Mat>& Dataset::val_images() const { return val_; }
const std::vector<cv::Mat>& Dataset::test_images() const { return test_; }

const std::vector<int>& Dataset::train_labels() const { return train_labels_; }
const std::vector<int>& Dataset::val_labels() const { return val_labels_; }
const std::vector<int>& Dataset::test_labels() const { return test_labels_; }

const std::unordered_map<std::string, int>& Dataset::class_mapping() const { return class_to_label_; }

std::string Dataset::class_name_for(int label) const {
    if (label_to_class_.contains(label)) {
        return label_to_class_.at(label);
    }
    return "<unknown>";
}

void Dataset::set_shuffle_seed(unsigned int seed) {
    shuffle_seed_ = seed;
}

} // namespace improc::ml
