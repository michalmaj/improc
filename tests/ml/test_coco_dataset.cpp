// tests/ml/test_coco_dataset.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
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

// ── ParseCocoJsonTest fixture ────────────────────────────────────────────────

class ParseCocoJsonTest : public testing::Test {
protected:
    static fs::path kCoco;

    static void SetUpTestSuite() {
        kCoco = setup_coco_sample();
    }
    static void TearDownTestSuite() {
        fs::remove_all(kCoco);
    }
};

fs::path ParseCocoJsonTest::kCoco;

// ── parse_coco_json ─────────────────────────────────────────────────────────

TEST_F(ParseCocoJsonTest, ValidJsonLoadsCorrectImageCount) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(kCoco / "annotations_train.json",
                             kCoco / "images", cm);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->size(), 2u);  // two images in train JSON
}

TEST_F(ParseCocoJsonTest, BBoxCoordinatesCorrect) {
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

TEST_F(ParseCocoJsonTest, NonContiguousCocoIdsRemappedTo0Indexed) {
    // cat=3, dog=7 in COCO → should be remapped to 0 and 1
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(kCoco / "annotations_train.json",
                             kCoco / "images", cm);
    ASSERT_TRUE(r.has_value());
    EXPECT_TRUE(cm.contains("cat"));
    EXPECT_TRUE(cm.contains("dog"));
    EXPECT_EQ(cm["cat"] + cm["dog"], 1);  // one is 0, the other is 1
}

TEST_F(ParseCocoJsonTest, CrowdAnnotationSkippedByDefault) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(kCoco / "annotations_val.json",
                             kCoco / "images", cm);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->size(), 1u);
    EXPECT_EQ(r->at(0).boxes.size(), 0u);  // iscrowd=1, skipped
}

TEST_F(ParseCocoJsonTest, CrowdAnnotationKeptWhenFlagFalse) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(kCoco / "annotations_val.json",
                             kCoco / "images", cm, /*skip_crowd=*/false);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->size(), 1u);
    EXPECT_EQ(r->at(0).boxes.size(), 1u);
    EXPECT_EQ(r->at(0).boxes[0].label, "dog");
}

TEST_F(ParseCocoJsonTest, MissingFileReturnsError) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(kCoco / "nonexistent.json",
                             kCoco / "images", cm);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::CocoJsonParseFailed);
}

TEST_F(ParseCocoJsonTest, MalformedJsonReturnsError) {
    auto bad = fs::temp_directory_path() / "improc_test_bad_coco.json";
    std::ofstream(bad) << "{ not valid json !!!";
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(bad, kCoco / "images", cm);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::CocoJsonParseFailed);
    fs::remove(bad);
}

TEST_F(ParseCocoJsonTest, MissingRequiredKeyReturnsError) {
    auto bad = fs::temp_directory_path() / "improc_test_noann.json";
    std::ofstream(bad) << R"({"images": [], "categories": []})";  // no "annotations"
    std::unordered_map<std::string, int> cm;
    auto r = parse_coco_json(bad, kCoco / "images", cm);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::CocoJsonParseFailed);
    fs::remove(bad);
}

TEST_F(ParseCocoJsonTest, FilterUnknownDropsObjectsNotInMap) {
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

// ── CocoDatasetTest fixture ──────────────────────────────────────────────────
//
// Uses its own temp directory ("improc_test_coco_ds") to avoid teardown
// conflicts with ParseCocoJsonTest which manages "improc_test_coco_sample".

class CocoDatasetTest : public testing::Test {
protected:
    static fs::path kCoco;

    static void SetUpTestSuite() {
        // setup_coco_sample() always recreates the directory from scratch.
        // We redirect it by temporarily pointing at our own name via a
        // lambda — actually we just call setup_coco_sample() and copy the
        // result, as the function hard-codes the dir name. Instead, duplicate
        // the fixture inline with a different root name.
        auto root = fs::temp_directory_path() / "improc_test_coco_ds";
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

        kCoco = root;
    }

    static void TearDownTestSuite() {
        fs::remove_all(kCoco);
    }
};

fs::path CocoDatasetTest::kCoco;

// ── CocoDataset tests ────────────────────────────────────────────────────────

TEST_F(CocoDatasetTest, LoadTrainSizesCorrect) {
    CocoDataset ds;
    auto r = ds.load_train(kCoco / "annotations_train.json", kCoco / "images");
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(ds.train().size(), 2u);
}

TEST_F(CocoDatasetTest, LoadValSizesCorrect) {
    CocoDataset ds;
    // skip_crowd(true) is the default
    auto r = ds.load_val(kCoco / "annotations_val.json", kCoco / "images");
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(ds.val().size(), 1u);
    EXPECT_TRUE(ds.val()[0].boxes.empty());  // iscrowd=1 skipped
}

TEST_F(CocoDatasetTest, LoadTrainAndValShareClassMap) {
    CocoDataset ds;
    ASSERT_TRUE(ds.load_train(kCoco / "annotations_train.json", kCoco / "images").has_value());
    ASSERT_TRUE(ds.load_val(kCoco / "annotations_val.json",   kCoco / "images").has_value());
    const auto& cm = ds.class_mapping();
    EXPECT_TRUE(cm.contains("cat"));
    EXPECT_TRUE(cm.contains("dog"));
    // Train encounters cat first, so cat=0, dog=1
    EXPECT_EQ(cm.at("cat"), 0);
    EXPECT_EQ(cm.at("dog"), 1);
}

TEST_F(CocoDatasetTest, ClassMappingBuiltCorrectly) {
    CocoDataset ds;
    ASSERT_TRUE(ds.load_train(kCoco / "annotations_train.json", kCoco / "images").has_value());
    const auto& cm = ds.class_mapping();
    EXPECT_TRUE(cm.contains("cat"));
    EXPECT_TRUE(cm.contains("dog"));
    // The two ids must be distinct and together form {0, 1}
    EXPECT_NE(cm.at("cat"), cm.at("dog"));
    EXPECT_EQ(cm.at("cat") + cm.at("dog"), 1);  // one is 0, the other is 1
}

TEST_F(CocoDatasetTest, ClassNameForRoundTrips) {
    CocoDataset ds;
    ASSERT_TRUE(ds.load_train(kCoco / "annotations_train.json", kCoco / "images").has_value());
    const auto& cm = ds.class_mapping();
    EXPECT_EQ(ds.class_name_for(cm.at("cat")), "cat");
    EXPECT_EQ(ds.class_name_for(cm.at("dog")), "dog");
    EXPECT_THROW(ds.class_name_for(99), std::out_of_range);
}

TEST_F(CocoDatasetTest, LoadTrainTwiceReplacesSplit) {
    CocoDataset ds;
    ASSERT_TRUE(ds.load_train(kCoco / "annotations_train.json", kCoco / "images").has_value());
    EXPECT_EQ(ds.train().size(), 2u);
    // Second load must replace, not accumulate
    ASSERT_TRUE(ds.load_train(kCoco / "annotations_train.json", kCoco / "images").has_value());
    EXPECT_EQ(ds.train().size(), 2u);
}

TEST_F(CocoDatasetTest, TrainBoxesCorrect) {
    CocoDataset ds;
    ASSERT_TRUE(ds.load_train(kCoco / "annotations_train.json", kCoco / "images").has_value());
    // Sorted by img_id: train()[0] = 000001 (1 box), train()[1] = 000002 (2 boxes)
    ASSERT_EQ(ds.train().size(), 2u);
    EXPECT_EQ(ds.train()[0].boxes.size(), 1u);  // 000001: one cat box
    EXPECT_EQ(ds.train()[1].boxes.size(), 2u);  // 000002: one cat + one dog box
}

// ── CocoDatasetEdgeTest fixture ──────────────────────────────────────────────

class CocoDatasetEdgeTest : public testing::Test {
protected:
    fs::path tmp_;

    void SetUp() override {
        tmp_ = fs::temp_directory_path() / "improc_coco_edge";
        fs::remove_all(tmp_);
        fs::create_directories(tmp_ / "images");
    }
    void TearDown() override {
        fs::remove_all(tmp_);
    }
};

// 1. Missing JSON file returns an error with CocoJsonParseFailed code.
TEST_F(CocoDatasetEdgeTest, MissingJsonReturnsError) {
    CocoDataset ds;
    auto result = ds.load_train(tmp_ / "nonexistent.json", tmp_ / "images");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, improc::Error::Code::CocoJsonParseFailed);
}

// 2. When all annotations have iscrowd=1 and skip_crowd is true, boxes must be empty.
TEST_F(CocoDatasetEdgeTest, AllCrowdAnnotationsGivesEmptyBoxes) {
    write_png(tmp_ / "images" / "img1.png", 10, 10);
    write_json(tmp_ / "coco.json", R"({
  "images": [{"id": 1, "file_name": "img1.png"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                   "bbox": [0, 0, 10, 10], "iscrowd": 1}],
  "categories": [{"id": 1, "name": "cat"}]
})");

    CocoDataset ds;
    ds.skip_crowd(true);
    auto result = ds.load_train(tmp_ / "coco.json", tmp_ / "images");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(ds.train().size(), 1u);
    EXPECT_TRUE(ds.train()[0].boxes.empty());
}

// 3. Pre-seeding classes with only "cat" filters out all "dog" annotations.
TEST_F(CocoDatasetEdgeTest, UserClassesFiltersUnknowns) {
    write_png(tmp_ / "images" / "img1.png", 10, 10);
    write_json(tmp_ / "coco.json", R"({
  "images": [{"id": 1, "file_name": "img1.png"}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 0},
    {"id": 2, "image_id": 1, "category_id": 2, "bbox": [5, 5, 5, 5], "iscrowd": 0}
  ],
  "categories": [
    {"id": 1, "name": "cat"},
    {"id": 2, "name": "dog"}
  ]
})");

    CocoDataset ds;
    ds.classes({"cat"});
    auto result = ds.load_train(tmp_ / "coco.json", tmp_ / "images");
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(ds.train().size(), 1u);
    for (const auto& bb : ds.train()[0].boxes)
        EXPECT_EQ(bb.label, "cat");
}

// 4. User-specified class order overrides encounter order in the JSON.
TEST_F(CocoDatasetEdgeTest, UserClassesFixesIdOrder) {
    write_png(tmp_ / "images" / "img1.png", 10, 10);
    write_json(tmp_ / "coco.json", R"({
  "images": [{"id": 1, "file_name": "img1.png"}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": 0},
    {"id": 2, "image_id": 1, "category_id": 2, "bbox": [5, 5, 5, 5], "iscrowd": 0}
  ],
  "categories": [
    {"id": 1, "name": "cat"},
    {"id": 2, "name": "dog"}
  ]
})");

    CocoDataset ds;
    ds.classes({"dog", "cat"});  // dog first — overrides JSON encounter order
    auto result = ds.load_train(tmp_ / "coco.json", tmp_ / "images");
    ASSERT_TRUE(result.has_value());
    const auto& cm = ds.class_mapping();
    EXPECT_EQ(cm.at("dog"), 0);
    EXPECT_EQ(cm.at("cat"), 1);
}

// 5. An image with no annotations is still included but has empty boxes.
TEST_F(CocoDatasetEdgeTest, EmptyAnnotationsGivesEmptyBoxes) {
    write_png(tmp_ / "images" / "img1.png", 10, 10);
    write_json(tmp_ / "coco.json", R"({
  "images": [{"id": 1, "file_name": "img1.png"}],
  "annotations": [],
  "categories": [{"id": 1, "name": "cat"}]
})");

    CocoDataset ds;
    auto result = ds.load_train(tmp_ / "coco.json", tmp_ / "images");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(ds.train().size(), 1u);
    EXPECT_TRUE(ds.train()[0].boxes.empty());
}
