// tests/ml/test_dnn_segmentor.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include "improc/ml/dnn_segmentor.hpp"
#include "improc/exceptions.hpp"

using namespace improc::ml;
namespace fs = std::filesystem;

// Reuse the tiny ONNX model created for OnnxSegmentor — cv::dnn can read it
// when OpenCV is built with Protobuf support.
static const fs::path TINY_ONNX =
    fs::path{__FILE__}.parent_path().parent_path()
    / "onnx" / "testdata" / "tiny_segmentor_semantic.onnx";

static Image<BGR> make_img(int rows = 8, int cols = 8) {
    return Image<BGR>(cv::Mat(rows, cols, CV_8UC3, cv::Scalar(100, 150, 200)));
}

// Helper: attempt to load TINY_ONNX; skip the test if cv::dnn lacks Protobuf support.
static std::optional<DnnSegmentor> try_load_segmentor(int w = 8, int h = 8) {
    try {
        return DnnSegmentor{TINY_ONNX}.input_size(w, h);
    } catch (const improc::ModelError& e) {
        std::string msg = e.what();
        if (msg.find("Protobuf") != std::string::npos ||
            msg.find("protobuf") != std::string::npos ||
            msg.find("onnx_importer") != std::string::npos) {
            return std::nullopt;   // cv::dnn built without Protobuf — skip inference tests
        }
        throw;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Error-path tests
// ──────────────────────────────────────────────────────────────────────────────

TEST(DnnSegmentorTest, NonExistentPathThrows) {
    EXPECT_THROW(DnnSegmentor{"nonexistent/model.onnx"}, improc::ModelError);
}

TEST(DnnSegmentorTest, CorruptModelThrows) {
    fs::path p = fs::temp_directory_path() / "improc_dnn_seg_dummy.onnx";
    { std::ofstream f(p); f << "not a real onnx"; }
    EXPECT_THROW(DnnSegmentor{p.string()}, improc::ModelError);
    fs::remove(p);
}

TEST(DnnSegmentorTest, ZeroScaleThrows) {
    auto seg = try_load_segmentor();
    if (!seg) GTEST_SKIP() << "cv::dnn ONNX support not available";
    EXPECT_THROW(seg->scale(0.0f), improc::ParameterError);
}

TEST(DnnSegmentorTest, NegativeScaleThrows) {
    auto seg = try_load_segmentor();
    if (!seg) GTEST_SKIP() << "cv::dnn ONNX support not available";
    EXPECT_THROW(seg->scale(-1.0f), improc::ParameterError);
}

TEST(DnnSegmentorTest, ZeroInputWidthThrows) {
    auto seg = try_load_segmentor();
    if (!seg) GTEST_SKIP() << "cv::dnn ONNX support not available";
    EXPECT_THROW(seg->input_size(0, 8), improc::ParameterError);
}

TEST(DnnSegmentorTest, ZeroInputHeightThrows) {
    auto seg = try_load_segmentor();
    if (!seg) GTEST_SKIP() << "cv::dnn ONNX support not available";
    EXPECT_THROW(seg->input_size(8, 0), improc::ParameterError);
}

// ──────────────────────────────────────────────────────────────────────────────
// Inference tests — skipped when cv::dnn was built without Protobuf support
// ──────────────────────────────────────────────────────────────────────────────

TEST(DnnSegmentorTest, InferenceReturnsExpected) {
    auto seg = try_load_segmentor();
    if (!seg) GTEST_SKIP() << "cv::dnn ONNX support not available (built without Protobuf)";
    auto result = (*seg)(make_img());
    EXPECT_TRUE(result.has_value()) << result.error().message;
}

TEST(DnnSegmentorTest, MaskSizeMatchesInput) {
    auto seg = try_load_segmentor();
    if (!seg) GTEST_SKIP() << "cv::dnn ONNX support not available (built without Protobuf)";
    auto img    = make_img(200, 200);
    auto result = (*seg)(img);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->class_mask.rows(), img.rows());
    EXPECT_EQ(result->class_mask.cols(), img.cols());
}

TEST(DnnSegmentorTest, MaskPixelsInClassRange) {
    auto seg = try_load_segmentor();
    if (!seg) GTEST_SKIP() << "cv::dnn ONNX support not available (built without Protobuf)";
    auto result = (*seg)(make_img());
    ASSERT_TRUE(result.has_value());
    double minVal, maxVal;
    cv::minMaxLoc(result->class_mask.mat(), &minVal, &maxVal);
    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 3.0);   // 4-class model -> max class id = 3
}

TEST(DnnSegmentorTest, LabelAtReturnsCorrectLabel) {
    auto seg = try_load_segmentor();
    if (!seg) GTEST_SKIP() << "cv::dnn ONNX support not available (built without Protobuf)";
    auto result = DnnSegmentor{TINY_ONNX}
        .input_size(8, 8)
        .labels({"bg", "cat", "dog", "bird"})
        (make_img());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->label_at(1), "cat");
}

TEST(DnnSegmentorTest, LabelAtOutOfRangeReturnsEmpty) {
    auto seg = try_load_segmentor();
    if (!seg) GTEST_SKIP() << "cv::dnn ONNX support not available (built without Protobuf)";
    auto result = (*seg)(make_img());
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->label_at(99), "");
}
