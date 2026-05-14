// tests/ml/test_coco_dataset.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <format>
#include <opencv2/imgcodecs.hpp>
#include "improc/ml/coco_dataset.hpp"
#include "improc/ml/annotated.hpp"

namespace fs = std::filesystem;
using namespace improc::ml;
using namespace improc::core;

// ── Test-data helpers ───────────────────────────────────────────────────────

static void write_png(const fs::path& path, int rows = 100, int cols = 100) {
    fs::create_directories(path.parent_path());
    cv::Mat mat(rows, cols, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::imwrite(path.string(), mat);
}

static void write_json(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << content;
}

// setup_coco_sample():
//   images/000001.png, 000002.png, 000003.png
//   annotations_train.json: images 1+2; cat(id=3) + dog(id=7) — non-contiguous ids
//   annotations_val.json: image 3; one dog annotation with iscrowd=1
static fs::path setup_coco_sample() {
    auto root = fs::temp_directory_path() / "improc_test_coco_sample";
    fs::remove_all(root);

    write_png(root / "images" / "000001.png");
    write_png(root / "images" / "000002.png");
    write_png(root / "images" / "000003.png");

    write_json(root / "annotations_train.json", R"({
  "images": [
    {"id": 1, "file_name": "000001.png"},
    {"id": 2, "file_name": "000002.png"}
  ],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10.0, 10.0, 40.0, 40.0], "iscrowd": 0},
    {"id": 2, "image_id": 2, "category_id": 3, "bbox": [5.0, 5.0, 40.0, 40.0],   "iscrowd": 0},
    {"id": 3, "image_id": 2, "category_id": 7, "bbox": [55.0, 55.0, 40.0, 40.0], "iscrowd": 0}
  ],
  "categories": [
    {"id": 3, "name": "cat"},
    {"id": 7, "name": "dog"}
  ]
})");

    write_json(root / "annotations_val.json", R"({
  "images": [
    {"id": 3, "file_name": "000003.png"}
  ],
  "annotations": [
    {"id": 4, "image_id": 3, "category_id": 7, "bbox": [20.0, 20.0, 60.0, 60.0], "iscrowd": 1}
  ],
  "categories": [
    {"id": 3, "name": "cat"},
    {"id": 7, "name": "dog"}
  ]
})");

    return root;
}

static const fs::path kCoco = setup_coco_sample();

// ── parse_coco_json ─────────────────────────────────────────────────────────

TEST(ParseCocoJsonTest, ValidJsonLoadsCorrectImageCount) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(kCoco / "annotations_train.json",
                             kCoco / "images", cm);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->size(), 2u);  // two images in train JSON
}

TEST(ParseCocoJsonTest, BBoxCoordinatesCorrect) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(kCoco / "annotations_train.json",
                             kCoco / "images", cm);
    ASSERT_TRUE(r.has_value());
    // Find the image with one box (image 000001 has only cat annotation id=1)
    const auto* single = [&]() -> const AnnotatedImage<BGR>* {
        for (const auto& ann : *r)
            if (ann.boxes.size() == 1u) return &ann;
        return nullptr;
    }();
    ASSERT_NE(single, nullptr);
    EXPECT_FLOAT_EQ(single->boxes[0].box.x,      10.f);
    EXPECT_FLOAT_EQ(single->boxes[0].box.y,      10.f);
    EXPECT_FLOAT_EQ(single->boxes[0].box.width,  40.f);
    EXPECT_FLOAT_EQ(single->boxes[0].box.height, 40.f);
}

TEST(ParseCocoJsonTest, NonContiguousCocoIdsRemappedTo0Indexed) {
    // cat=3, dog=7 in COCO → should be remapped to 0 and 1
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(kCoco / "annotations_train.json",
                             kCoco / "images", cm);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(cm.contains("cat"));
    EXPECT_TRUE(cm.contains("dog"));
    EXPECT_EQ(cm["cat"] + cm["dog"], 1);  // one is 0, the other is 1
}

TEST(ParseCocoJsonTest, CrowdAnnotationSkippedByDefault) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(kCoco / "annotations_val.json",
                             kCoco / "images", cm);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->size(), 1u);
    EXPECT_EQ(r->at(0).boxes.size(), 0u);  // iscrowd=1, skipped
}

TEST(ParseCocoJsonTest, CrowdAnnotationKeptWhenFlagFalse) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(kCoco / "annotations_val.json",
                             kCoco / "images", cm, /*skip_crowd=*/false);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->size(), 1u);
    EXPECT_EQ(r->at(0).boxes.size(), 1u);
    EXPECT_EQ(r->at(0).boxes[0].label, "dog");
}

TEST(ParseCocoJsonTest, MissingFileReturnsError) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(kCoco / "nonexistent.json",
                             kCoco / "images", cm);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::CocoJsonParseFailed);
}

TEST(ParseCocoJsonTest, MalformedJsonReturnsError) {
    auto bad = fs::temp_directory_path() / "improc_test_bad_coco.json";
    std::ofstream(bad) << "{ not valid json !!!";
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(bad, kCoco / "images", cm);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::CocoJsonParseFailed);
    fs::remove(bad);
}

TEST(ParseCocoJsonTest, MissingRequiredKeyReturnsError) {
    auto bad = fs::temp_directory_path() / "improc_test_noann.json";
    std::ofstream(bad) << R"({"images": [], "categories": []})";  // no "annotations"
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(bad, kCoco / "images", cm);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::CocoJsonParseFailed);
    fs::remove(bad);
}

TEST(ParseCocoJsonTest, FilterUnknownDropsObjectsNotInMap) {
    std::unordered_map<std::string, int> cm{{"cat", 0}};  // dog not in map
    auto r = parse_coco_json(kCoco / "annotations_train.json",
                             kCoco / "images", cm,
                             /*skip_crowd=*/true, /*filter_unknown=*/true);
    ASSERT_TRUE(r.has_value());
    // All boxes should be "cat" only
    for (const auto& ann : *r)
        for (const auto& bb : ann.boxes)
            EXPECT_EQ(bb.label, "cat");
    EXPECT_EQ(cm.size(), 1u);  // dog NOT added
}
