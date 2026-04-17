// Created by Michał Maj on 14/04/2025.

#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include <filesystem>
#include <fstream>
#include "improc/ml/haar_cascade_loader.hpp"

using improc::ml::HaarCascadeLoader;

TEST(HaarCascadeLoaderTest, CanLoadValidCascade) {
  HaarCascadeLoader loader;
  auto path = std::filesystem::path(__FILE__).parent_path() / "testdata" / "haarcascade_frontalface_default.xml";

  ASSERT_NO_THROW(loader.load_model(path));

  auto model = loader.get_model();
  EXPECT_TRUE(model.has_value());
  EXPECT_FALSE(model->empty());
}

TEST(HaarCascadeLoaderTest, InvalidPathReturnsError) {
  HaarCascadeLoader loader;
  auto result = HaarCascadeLoader::check_path("not/a/real/path.xml");
  EXPECT_FALSE(result.has_value());
}

TEST(HaarCascadeLoaderTest, ThrowsOnInvalidLoad) {
  HaarCascadeLoader loader;
  auto path = std::filesystem::path(__FILE__).parent_path() / "invalid_file.txt";
  std::ofstream(path) << "not a valid model";

  EXPECT_THROW(loader.load_model(path), improc::ModelError);
  std::filesystem::remove(path);
}
