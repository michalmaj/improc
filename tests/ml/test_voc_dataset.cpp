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
