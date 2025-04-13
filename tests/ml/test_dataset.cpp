//
// Created by Michał Maj on 13/04/2025.
//

#include "../../include/improc/ml/dataset.hpp"
#include <filesystem>
#include <gtest/gtest.h>

using improc::ml::Dataset;

TEST(DatasetTest, LoadAndSplitFromValidPath) {
  Dataset dataset;
  auto test_dir = std::filesystem::path(__FILE__).parent_path() / "glasses_dataset";

  auto result = dataset.load_from_directory(test_dir, 0.2f, 0.1f);
  ASSERT_TRUE(result.has_value());

  size_t total = dataset.train_images().size() +
                 dataset.val_images().size() +
                 dataset.test_images().size();

  EXPECT_GT(total, 0);
  EXPECT_EQ(dataset.train_labels().size(), dataset.train_images().size());
  EXPECT_EQ(dataset.test_labels().size(), dataset.test_images().size());
  EXPECT_EQ(dataset.val_labels().size(), dataset.val_images().size());
}

TEST(DatasetTest, ClassNameMappingWorks) {
  Dataset dataset;
  auto test_dir = std::filesystem::path(__FILE__).parent_path() / "glasses_dataset";

  auto result = dataset.load_from_directory(test_dir);
  ASSERT_TRUE(result.has_value());

  for (const auto& [name, label] : dataset.class_mapping()) {
    EXPECT_EQ(dataset.class_name_for(label), name);
  }
}

TEST(DatasetTest, HandlesInvalidPath) {
  Dataset dataset;
  auto result = dataset.load_from_directory("invalid/path");
  EXPECT_FALSE(result.has_value());
}
