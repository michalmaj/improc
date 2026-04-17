//
// Created by Michał Maj on 13/04/2025.
//

#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/ml/image_loader.hpp"

using improc::ml::ImageLoader;

TEST(ImageLoaderTest, InvalidPathThrows) {
  ImageLoader loader;
  EXPECT_THROW(loader.load_images("not_a_dir"), improc::FileNotFoundError);
}

TEST(ImageLoaderTest, EmptyDirReturnsNoImages) {
  ImageLoader loader;
  auto empty_dir = std::filesystem::path(__FILE__).parent_path() / "testdata" / "empty";
  loader.load_images(empty_dir);
  auto result = loader.get_images();
  EXPECT_TRUE(!result.has_value());
}

TEST(ImageLoaderTest, ValidImagesAreLoaded) {
  ImageLoader loader;
  auto test_dir = std::filesystem::path(__FILE__).parent_path() / "testdata" / "images";
  loader.load_images(test_dir);
  auto result = loader.get_images();
  EXPECT_TRUE(result.has_value());
  EXPECT_FALSE(result->empty());
}
