// tests/onnx/test_onnx_classifier.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include "improc/onnx/onnx_classifier.hpp"
#include "improc/exceptions.hpp"

using namespace improc::onnx;
using improc::core::Image;
using improc::core::BGR;
namespace fs = std::filesystem;

static const fs::path kTestData =
    fs::path{__FILE__}.parent_path() / "testdata";

// ── Construction / validation ─────────────────────────────────────────────────

TEST(OnnxClassifierTest, NonExistentPathThrows) {
    EXPECT_THROW(OnnxClassifier{"nonexistent/model.onnx"}, improc::ModelError);
}

TEST(OnnxClassifierTest, CorruptModelThrows) {
    fs::path p = fs::temp_directory_path() / "improc_corrupt_cls.onnx";
    { std::ofstream f(p); f << "not onnx"; }
    EXPECT_THROW(OnnxClassifier{p}, improc::ModelError);
    fs::remove(p);
}

TEST(OnnxClassifierTest, ZeroTopKThrows) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxClassifier{model}.top_k(0), improc::ParameterError);
}

TEST(OnnxClassifierTest, NegativeTopKThrows) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxClassifier{model}.top_k(-1), improc::ParameterError);
}

TEST(OnnxClassifierTest, ZeroScaleThrows) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxClassifier{model}.scale(0.0f), improc::ParameterError);
}

TEST(OnnxClassifierTest, NegativeScaleThrows) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxClassifier{model}.scale(-0.1f), improc::ParameterError);
}

TEST(OnnxClassifierTest, ZeroInputWidthThrows) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxClassifier{model}.input_size(0, 8), improc::ParameterError);
}

TEST(OnnxClassifierTest, ZeroInputHeightThrows) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxClassifier{model}.input_size(8, 0), improc::ParameterError);
}

// ── Inference ─────────────────────────────────────────────────────────────────

TEST(OnnxClassifierTest, InferenceReturnsTopKResults) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));

    // tiny_classifier expects 8×8 input
    OnnxClassifier cls{model};
    cls.input_size(8, 8).top_k(3).swap_rb(false);

    cv::Mat mat(16, 16, CV_8UC3, cv::Scalar(128, 64, 32));
    Image<BGR> img{mat};

    auto result = cls(img);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_LE(result.value().size(), 3u);
    EXPECT_FALSE(result.value().empty());

    // Results must be sorted descending by score
    const auto& res = result.value();
    for (size_t i = 1; i < res.size(); ++i)
        EXPECT_GE(res[i - 1].score, res[i].score);
}

TEST(OnnxClassifierTest, LabelsArePopulated) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));

    std::vector<std::string> lbls{"cat", "dog", "bird"};
    OnnxClassifier cls{model};
    cls.input_size(8, 8).top_k(1).labels(lbls).swap_rb(false);

    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(100));
    Image<BGR> img{mat};

    auto result = cls(img);
    ASSERT_TRUE(result.has_value());
    ASSERT_FALSE(result.value().empty());
    EXPECT_FALSE(result.value()[0].label.empty());
}

TEST(OnnxClassifierTest, TopKCappedByNumClasses) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));

    // Model has 3 classes; requesting top_k=10 should give at most 3
    OnnxClassifier cls{model};
    cls.input_size(8, 8).top_k(10).swap_rb(false);

    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(50));
    Image<BGR> img{mat};

    auto result = cls(img);
    ASSERT_TRUE(result.has_value());
    EXPECT_LE(result.value().size(), 3u);
}
