// tests/ml/test_voc_seg_dataset.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include "improc/ml/voc_seg_dataset.hpp"
#include "improc/error.hpp"

namespace fs = std::filesystem;
using namespace improc::ml;

static void write_mask_png(const fs::path& p, int rows, int cols, uint8_t fill) {
    cv::Mat m(rows, cols, CV_8UC1, cv::Scalar(fill));
    cv::imwrite(p.string(), m);
}

static void write_bgr_png(const fs::path& p, int rows, int cols) {
    cv::Mat m(rows, cols, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::imwrite(p.string(), m);
}

static void write_txt(const fs::path& p, const std::string& content) {
    std::ofstream f(p); f << content;
}

static fs::path setup_voc_seg() {
    fs::path root = fs::temp_directory_path() / "improc_test_voc_seg";
    fs::remove_all(root);
    fs::create_directories(root / "JPEGImages");
    fs::create_directories(root / "SegmentationClass");
    fs::create_directories(root / "SegmentationObject");
    fs::create_directories(root / "ImageSets" / "Segmentation");

    write_bgr_png(root / "JPEGImages" / "img1.png", 10, 10);
    write_bgr_png(root / "JPEGImages" / "img2.png", 10, 10);
    write_mask_png(root / "SegmentationClass" / "img1.png", 10, 10, 1);
    write_mask_png(root / "SegmentationClass" / "img2.png", 10, 10, 2);
    write_mask_png(root / "SegmentationObject" / "img1.png", 10, 10, 10);
    write_mask_png(root / "SegmentationObject" / "img2.png", 10, 10, 20);
    write_txt(root / "ImageSets" / "Segmentation" / "train.txt", "img1\n");
    write_txt(root / "ImageSets" / "Segmentation" / "val.txt",   "img2\n");
    return root;
}

class ParseVocSegTest : public testing::Test {
protected:
    static fs::path root_;
    static void SetUpTestSuite()    { root_ = setup_voc_seg(); }
    static void TearDownTestSuite() { fs::remove_all(root_); }
};
fs::path ParseVocSegTest::root_;

TEST_F(ParseVocSegTest, ValidParseReturnsCorrectSizes) {
    auto r = parse_voc_seg("img1",
                           root_ / "JPEGImages",
                           root_ / "SegmentationClass",
                           /*instance_masks_dir=*/{});
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->image.rows(), 10);
    EXPECT_EQ(r->class_mask.rows(), 10);
    EXPECT_FALSE(r->instance_mask.has_value());
}

TEST_F(ParseVocSegTest, ClassMaskValuesPreserved) {
    auto r = parse_voc_seg("img1",
                           root_ / "JPEGImages",
                           root_ / "SegmentationClass",
                           {});
    ASSERT_TRUE(r.has_value());
    int nonone = cv::countNonZero(r->class_mask.mat() != 1);
    EXPECT_EQ(nonone, 0);
}

TEST_F(ParseVocSegTest, InstanceMaskLoadedWhenDirProvided) {
    auto r = parse_voc_seg("img1",
                           root_ / "JPEGImages",
                           root_ / "SegmentationClass",
                           root_ / "SegmentationObject");
    ASSERT_TRUE(r.has_value());
    ASSERT_TRUE(r->instance_mask.has_value());
    int nonton = cv::countNonZero(r->instance_mask->mat() != 10);
    EXPECT_EQ(nonton, 0);
}

TEST_F(ParseVocSegTest, MissingImageReturnsError) {
    auto r = parse_voc_seg("nonexistent",
                           root_ / "JPEGImages",
                           root_ / "SegmentationClass",
                           {});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::VocSegParseFailed);
}

TEST_F(ParseVocSegTest, MissingClassMaskReturnsError) {
    auto r = parse_voc_seg("img1",
                           root_ / "JPEGImages",
                           root_ / "nonexistent_masks",
                           {});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::VocSegParseFailed);
}

TEST_F(ParseVocSegTest, PngImageFallbackWorks) {
    // img1 is stored as .png; the function tries .jpg first, falls back to .png
    auto r = parse_voc_seg("img1",
                           root_ / "JPEGImages",
                           root_ / "SegmentationClass",
                           {});
    ASSERT_TRUE(r.has_value());  // .png fallback succeeded
}

static fs::path setup_voc_seg_ds() {
    // Separate tmpdir from ParseVocSegTest to avoid teardown conflicts
    fs::path root = fs::temp_directory_path() / "improc_test_voc_seg_ds";
    fs::remove_all(root);
    fs::create_directories(root / "JPEGImages");
    fs::create_directories(root / "SegmentationClass");
    fs::create_directories(root / "SegmentationObject");
    fs::create_directories(root / "ImageSets" / "Segmentation");
    write_bgr_png(root / "JPEGImages" / "img1.png", 10, 10);
    write_bgr_png(root / "JPEGImages" / "img2.png", 10, 10);
    write_mask_png(root / "SegmentationClass" / "img1.png", 10, 10, 1);
    write_mask_png(root / "SegmentationClass" / "img2.png", 10, 10, 2);
    write_mask_png(root / "SegmentationObject" / "img1.png", 10, 10, 10);
    write_mask_png(root / "SegmentationObject" / "img2.png", 10, 10, 20);
    write_txt(root / "ImageSets" / "Segmentation" / "train.txt", "img1\n");
    write_txt(root / "ImageSets" / "Segmentation" / "val.txt",   "img2\n");
    return root;
}

class VocSegDatasetTest : public testing::Test {
protected:
    static fs::path root_;
    static void SetUpTestSuite()    { root_ = setup_voc_seg_ds(); }
    static void TearDownTestSuite() { fs::remove_all(root_); }
};
fs::path VocSegDatasetTest::root_;

TEST_F(VocSegDatasetTest, LoadTrainAndValSizes) {
    VocSegDataset ds;
    auto r = ds.load_from_directory(root_);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(ds.train().size(), 1u);
    EXPECT_EQ(ds.val().size(),   1u);
    EXPECT_EQ(ds.test().size(),  0u);
}

TEST_F(VocSegDatasetTest, TrainImageAndMaskCorrect) {
    VocSegDataset ds;
    ASSERT_TRUE(ds.load_from_directory(root_).has_value());
    EXPECT_EQ(ds.train()[0].image.rows(), 10);
    EXPECT_EQ(ds.train()[0].class_mask.rows(), 10);
    int nonnone = cv::countNonZero(ds.train()[0].class_mask.mat() != 1);
    EXPECT_EQ(nonnone, 0);
}

TEST_F(VocSegDatasetTest, InstanceMaskNulloptByDefault) {
    VocSegDataset ds;
    ASSERT_TRUE(ds.load_from_directory(root_).has_value());
    EXPECT_FALSE(ds.train()[0].instance_mask.has_value());
}

TEST_F(VocSegDatasetTest, LoadInstanceMasksPopulatesOptional) {
    VocSegDataset ds;
    ds.load_instance_masks(true);
    ASSERT_TRUE(ds.load_from_directory(root_).has_value());
    ASSERT_TRUE(ds.train()[0].instance_mask.has_value());
    int nonnone = cv::countNonZero(ds.train()[0].instance_mask->mat() != 10);
    EXPECT_EQ(nonnone, 0);
}

TEST_F(VocSegDatasetTest, ClassNameForRoundTrips) {
    VocSegDataset ds;
    ds.classes({"background", "cat", "dog"});
    ASSERT_TRUE(ds.load_from_directory(root_).has_value());
    EXPECT_EQ(ds.class_name_for(0), "background");
    EXPECT_EQ(ds.class_name_for(1), "cat");
    EXPECT_EQ(ds.class_name_for(2), "dog");
}

TEST_F(VocSegDatasetTest, ClassNameForThrowsWithoutClasses) {
    VocSegDataset ds;
    ASSERT_TRUE(ds.load_from_directory(root_).has_value());
    EXPECT_THROW(ds.class_name_for(1), std::out_of_range);
}

TEST_F(VocSegDatasetTest, MissingSegClassDirReturnsError) {
    fs::path bad = fs::temp_directory_path() / "improc_voc_seg_nodir";
    fs::remove_all(bad);
    fs::create_directories(bad / "JPEGImages");
    VocSegDataset ds;
    auto r = ds.load_from_directory(bad);
    EXPECT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::VocSegParseFailed);
    fs::remove_all(bad);
}

TEST_F(VocSegDatasetTest, RandomSplitFallback) {
    fs::path root = fs::temp_directory_path() / "improc_voc_seg_rand";
    fs::remove_all(root);
    fs::create_directories(root / "JPEGImages");
    fs::create_directories(root / "SegmentationClass");
    for (int i = 1; i <= 5; ++i) {
        std::string s = "img" + std::to_string(i);
        write_bgr_png(root / "JPEGImages" / (s + ".png"), 10, 10);
        write_mask_png(root / "SegmentationClass" / (s + ".png"), 10, 10, static_cast<uint8_t>(i));
    }
    VocSegDataset ds;
    auto r = ds.load_from_directory(root);
    ASSERT_TRUE(r.has_value());
    EXPECT_GT(ds.train().size(), 0u);
    EXPECT_EQ(ds.train().size() + ds.val().size() + ds.test().size(), 5u);
    fs::remove_all(root);
}
