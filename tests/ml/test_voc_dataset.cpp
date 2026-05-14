// tests/ml/test_voc_dataset.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include "improc/ml/voc_dataset.hpp"
#include "improc/ml/annotated.hpp"

namespace fs = std::filesystem;
using namespace improc::ml;
using namespace improc::core;

// ── Test-data helpers ───────────────────────────────────────────────────────

static void write_xml(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << content;
}

static void write_png(const fs::path& path, int rows = 100, int cols = 100) {
    fs::create_directories(path.parent_path());
    cv::Mat mat(rows, cols, CV_8UC3, cv::Scalar(100, 150, 200));
    cv::imwrite(path.string(), mat);
}

static void write_txt(const fs::path& path, const std::string& content) {
    fs::create_directories(path.parent_path());
    std::ofstream(path) << content;
}

// voc_sample: 3 images, VOC split (train=000001+000002, val=000003)
static fs::path setup_voc_sample() {
    auto root = fs::temp_directory_path() / "improc_test_voc_sample";
    fs::remove_all(root);

    write_png(root / "JPEGImages" / "000001.png");
    write_png(root / "JPEGImages" / "000002.png");
    write_png(root / "JPEGImages" / "000003.png");

    write_xml(root / "Annotations" / "000001.xml", R"(
<annotation>
    <filename>000001.png</filename>
    <size><width>100</width><height>100</height><depth>3</depth></size>
    <object>
        <name>cat</name><difficult>0</difficult>
        <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
    </object>
</annotation>)");

    write_xml(root / "Annotations" / "000002.xml", R"(
<annotation>
    <filename>000002.png</filename>
    <size><width>100</width><height>100</height><depth>3</depth></size>
    <object>
        <name>cat</name><difficult>0</difficult>
        <bndbox><xmin>5</xmin><ymin>5</ymin><xmax>45</xmax><ymax>45</ymax></bndbox>
    </object>
    <object>
        <name>dog</name><difficult>0</difficult>
        <bndbox><xmin>55</xmin><ymin>55</ymin><xmax>95</xmax><ymax>95</ymax></bndbox>
    </object>
</annotation>)");

    write_xml(root / "Annotations" / "000003.xml", R"(
<annotation>
    <filename>000003.png</filename>
    <size><width>100</width><height>100</height><depth>3</depth></size>
    <object>
        <name>dog</name><difficult>1</difficult>
        <bndbox><xmin>20</xmin><ymin>20</ymin><xmax>80</xmax><ymax>80</ymax></bndbox>
    </object>
</annotation>)");

    write_txt(root / "ImageSets" / "Main" / "train.txt", "000001\n000002\n");
    write_txt(root / "ImageSets" / "Main" / "val.txt",   "000003\n");
    write_txt(root / "ImageSets" / "Main" / "test.txt",  "");

    return root;
}

// voc_sample_nosplit: 5 images, no ImageSets directory
static fs::path setup_voc_nosplit() {
    auto root = fs::temp_directory_path() / "improc_test_voc_nosplit";
    fs::remove_all(root);

    for (int i = 1; i <= 5; ++i) {
        auto stem = std::format("{:06d}", i);
        write_png(root / "JPEGImages" / (stem + ".png"));
        std::string cls = (i % 2 == 0) ? "dog" : "cat";
        write_xml(root / "Annotations" / (stem + ".xml"),
            "<annotation><filename>" + stem + ".png</filename>"
            "<size><width>100</width><height>100</height><depth>3</depth></size>"
            "<object><name>" + cls + "</name><difficult>0</difficult>"
            "<bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>"
            "</object></annotation>");
    }
    return root;
}

static const fs::path kVocSample  = setup_voc_sample();
static const fs::path kVocNoSplit = setup_voc_nosplit();

// ── parse_voc_xml ───────────────────────────────────────────────────────────

TEST(ParseVocXmlTest, ValidXmlLoadsSingleObject) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_voc_xml(kVocSample / "Annotations" / "000001.xml",
                           kVocSample / "JPEGImages", cm);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->image.rows(), 100);
    EXPECT_EQ(r->image.cols(), 100);
    ASSERT_EQ(r->boxes.size(), 1u);
    EXPECT_EQ(r->boxes[0].label, "cat");
    EXPECT_EQ(r->boxes[0].class_id, 0);
    EXPECT_FLOAT_EQ(r->boxes[0].box.x,      10.f);
    EXPECT_FLOAT_EQ(r->boxes[0].box.y,      10.f);
    EXPECT_FLOAT_EQ(r->boxes[0].box.width,  40.f);   // xmax-xmin = 50-10
    EXPECT_FLOAT_EQ(r->boxes[0].box.height, 40.f);   // ymax-ymin = 50-10
}

TEST(ParseVocXmlTest, ValidXmlLoadsMultipleObjects) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_voc_xml(kVocSample / "Annotations" / "000002.xml",
                           kVocSample / "JPEGImages", cm);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->boxes.size(), 2u);
    EXPECT_EQ(r->boxes[0].label, "cat");
    EXPECT_EQ(r->boxes[1].label, "dog");
    EXPECT_EQ(cm.size(), 2u);
}

TEST(ParseVocXmlTest, DifficultObjectSkippedByDefault) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_voc_xml(kVocSample / "Annotations" / "000003.xml",
                           kVocSample / "JPEGImages", cm);
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(r->boxes.size(), 0u);   // difficult=1, skip_difficult=true
}

TEST(ParseVocXmlTest, DifficultObjectKeptWhenFlagFalse) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_voc_xml(kVocSample / "Annotations" / "000003.xml",
                           kVocSample / "JPEGImages", cm, /*skip_difficult=*/false);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->boxes.size(), 1u);
    EXPECT_EQ(r->boxes[0].label, "dog");
}

TEST(ParseVocXmlTest, MissingXmlReturnsError) {
    std::unordered_map<std::string, int> cm;
    auto r = parse_voc_xml(kVocSample / "Annotations" / "nonexistent.xml",
                           kVocSample / "JPEGImages", cm);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::VocXmlParseFailed);
}

TEST(ParseVocXmlTest, MissingImageReturnsError) {
    // Write an XML pointing to a non-existent image
    auto xml_path = fs::temp_directory_path() / "improc_test_bad_img.xml";
    std::ofstream(xml_path) <<
        "<annotation><filename>no_such.png</filename>"
        "<size><width>100</width><height>100</height><depth>3</depth></size>"
        "</annotation>";
    std::unordered_map<std::string, int> cm;
    auto r = parse_voc_xml(xml_path, kVocSample / "JPEGImages", cm);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::ImageReadFailed);
    fs::remove(xml_path);
}

TEST(ParseVocXmlTest, FilterUnknownDropsObjectsNotInMap) {
    std::unordered_map<std::string, int> cm{{"cat", 0}};  // only cat
    auto r = parse_voc_xml(kVocSample / "Annotations" / "000002.xml",
                           kVocSample / "JPEGImages", cm,
                           /*skip_difficult=*/true, /*filter_unknown=*/true);
    ASSERT_TRUE(r.has_value());
    ASSERT_EQ(r->boxes.size(), 1u);   // dog dropped, cat kept
    EXPECT_EQ(r->boxes[0].label, "cat");
    EXPECT_EQ(cm.size(), 1u);         // dog NOT added to map
}

// ── VocDataset — VOC split mode ─────────────────────────────────────────────

TEST(VocDatasetSplitTest, TrainValSizeMatchSplitFiles) {
    VocDataset ds;
    ASSERT_TRUE(ds.load_from_directory(kVocSample).has_value());
    EXPECT_EQ(ds.train().size(), 2u);  // 000001, 000002
    EXPECT_EQ(ds.val().size(),   1u);  // 000003
    EXPECT_EQ(ds.test().size(),  0u);  // test.txt is empty
}

TEST(VocDatasetSplitTest, ClassMappingBuiltCorrectly) {
    VocDataset ds;
    ASSERT_TRUE(ds.load_from_directory(kVocSample).has_value());
    const auto& cm = ds.class_mapping();
    EXPECT_TRUE(cm.contains("cat"));
    EXPECT_TRUE(cm.contains("dog"));
    EXPECT_EQ(cm.size(), 2u);
}

TEST(VocDatasetSplitTest, ClassNameForRoundTrips) {
    VocDataset ds;
    ASSERT_TRUE(ds.load_from_directory(kVocSample).has_value());
    const auto& cm = ds.class_mapping();
    EXPECT_EQ(ds.class_name_for(cm.at("cat")), "cat");
    EXPECT_EQ(ds.class_name_for(cm.at("dog")), "dog");
}

TEST(VocDatasetSplitTest, DifficultObjectSkippedInVocSplit) {
    // 000003 has one dog marked difficult=1, skip_difficult=true by default
    VocDataset ds;
    ASSERT_TRUE(ds.load_from_directory(kVocSample).has_value());
    ASSERT_EQ(ds.val().size(), 1u);
    EXPECT_EQ(ds.val()[0].boxes.size(), 0u);
}

TEST(VocDatasetSplitTest, DifficultKeptWhenFlagFalse) {
    VocDataset ds;
    ds.skip_difficult(false);
    ASSERT_TRUE(ds.load_from_directory(kVocSample).has_value());
    ASSERT_EQ(ds.val().size(), 1u);
    EXPECT_EQ(ds.val()[0].boxes.size(), 1u);
    EXPECT_EQ(ds.val()[0].boxes[0].label, "dog");
}

// ── VocDataset — random split mode ─────────────────────────────────────────

TEST(VocDatasetRandomSplitTest, SumOfSplitsEqualsTotal) {
    VocDataset ds;
    ds.shuffle_seed(42).test_ratio(0.2f).val_ratio(0.2f);
    ASSERT_TRUE(ds.load_from_directory(kVocNoSplit).has_value());
    EXPECT_EQ(ds.train().size() + ds.val().size() + ds.test().size(), 5u);
}

TEST(VocDatasetRandomSplitTest, SeedProducesDeterministicSplit) {
    VocDataset ds1, ds2;
    ds1.shuffle_seed(99).test_ratio(0.2f).val_ratio(0.2f);
    ds2.shuffle_seed(99).test_ratio(0.2f).val_ratio(0.2f);
    ASSERT_TRUE(ds1.load_from_directory(kVocNoSplit).has_value());
    ASSERT_TRUE(ds2.load_from_directory(kVocNoSplit).has_value());
    ASSERT_EQ(ds1.train().size(), ds2.train().size());
    EXPECT_EQ(cv::norm(ds1.train()[0].image.mat(),
                       ds2.train()[0].image.mat(), cv::NORM_INF), 0.0);
}

TEST(VocDatasetRandomSplitTest, RatiosRespected) {
    // 5 images, test=0.2→1, val=0.2→1, train=3
    VocDataset ds;
    ds.shuffle_seed(0).test_ratio(0.2f).val_ratio(0.2f);
    ASSERT_TRUE(ds.load_from_directory(kVocNoSplit).has_value());
    EXPECT_EQ(ds.train().size(), 3u);
    EXPECT_EQ(ds.val().size(),   1u);
    EXPECT_EQ(ds.test().size(),  1u);
}

// ── VocDataset — classes() filter ──────────────────────────────────────────

TEST(VocDatasetClassFilterTest, UnknownClassObjectsDropped) {
    VocDataset ds;
    ds.classes({"cat"});   // dog is not in list
    ASSERT_TRUE(ds.load_from_directory(kVocSample).has_value());
    EXPECT_FALSE(ds.class_mapping().contains("dog"));
    for (const auto& ann : ds.train())
        for (const auto& bb : ann.boxes)
            EXPECT_EQ(bb.label, "cat");
}

TEST(VocDatasetClassFilterTest, UserListFixesIdOrder) {
    VocDataset ds;
    ds.classes({"dog", "cat"});   // dog=0, cat=1
    ASSERT_TRUE(ds.load_from_directory(kVocSample).has_value());
    EXPECT_EQ(ds.class_mapping().at("dog"), 0);
    EXPECT_EQ(ds.class_mapping().at("cat"), 1);
}

TEST(VocDatasetClassFilterTest, ClassNameForUnknownThrows) {
    VocDataset ds;
    ASSERT_TRUE(ds.load_from_directory(kVocSample).has_value());
    EXPECT_THROW(ds.class_name_for(99), std::out_of_range);
}

// ── VocDataset — edge cases ─────────────────────────────────────────────────

TEST(VocDatasetEdgeTest, MissingRootReturnsError) {
    VocDataset ds;
    auto r = ds.load_from_directory("/nonexistent/path/voc");
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::DirectoryNotFound);
}

TEST(VocDatasetEdgeTest, MissingAnnotationsDirReturnsError) {
    auto root = fs::temp_directory_path() / "improc_test_voc_noanno";
    fs::remove_all(root);
    fs::create_directories(root / "JPEGImages");
    VocDataset ds;
    auto r = ds.load_from_directory(root);
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, improc::Error::Code::DirectoryNotFound);
    fs::remove_all(root);
}

TEST(VocDatasetEdgeTest, AllDifficultObjectsResultsInEmptyBoxes) {
    VocDataset ds;
    ds.skip_difficult(true);
    ASSERT_TRUE(ds.load_from_directory(kVocSample).has_value());
    ASSERT_EQ(ds.val().size(), 1u);
    EXPECT_TRUE(ds.val()[0].boxes.empty());
}

TEST(VocDatasetEdgeTest, MissingSplitFileGivesEmptyVector) {
    auto root = fs::temp_directory_path() / "improc_test_voc_nosplitfile";
    fs::remove_all(root);
    fs::copy(kVocSample, root, fs::copy_options::recursive);
    fs::remove(root / "ImageSets" / "Main" / "test.txt");
    VocDataset ds;
    ASSERT_TRUE(ds.load_from_directory(root).has_value());
    EXPECT_EQ(ds.test().size(), 0u);
    fs::remove_all(root);
}
