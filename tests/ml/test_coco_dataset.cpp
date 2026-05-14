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
