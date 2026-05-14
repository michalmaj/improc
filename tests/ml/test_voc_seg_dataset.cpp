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
