// tests/onnx/test_onnx_session.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "improc/onnx/onnx_session.hpp"

using namespace improc::onnx;
namespace fs = std::filesystem;

static const fs::path kTestData =
    fs::path{__FILE__}.parent_path() / "testdata";

// ── OnnxSession load ──────────────────────────────────────────────────────────

TEST(OnnxSessionTest, NotLoadedByDefault) {
    OnnxSession s;
    EXPECT_FALSE(s.is_loaded());
    EXPECT_TRUE(s.input_names().empty());
    EXPECT_TRUE(s.output_names().empty());
}

TEST(OnnxSessionTest, LoadMissingFileReturnsError) {
    OnnxSession s;
    auto result = s.load("nonexistent/model.onnx");
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, improc::Error::Code::InvalidModelFile);
}

TEST(OnnxSessionTest, LoadWrongExtensionReturnsError) {
    fs::path p = fs::temp_directory_path() / "improc_test_session.pb";
    { std::ofstream f(p); f << "not a model"; }
    OnnxSession s;
    auto result = s.load(p);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, improc::Error::Code::InvalidModelFile);
    fs::remove(p);
}

TEST(OnnxSessionTest, LoadCorruptFileReturnsError) {
    fs::path p = fs::temp_directory_path() / "improc_test_corrupt.onnx";
    { std::ofstream f(p); f << "definitely not onnx"; }
    OnnxSession s;
    auto result = s.load(p);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, improc::Error::Code::OnnxModelLoadFailed);
    fs::remove(p);
}

TEST(OnnxSessionTest, RunBeforeLoadReturnsError) {
    OnnxSession s;
    auto result = s.run({});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, improc::Error::Code::OnnxSessionNotLoaded);
}

// ── OnnxSession inference with tiny_classifier.onnx ──────────────────────────

TEST(OnnxSessionTest, LoadValidModelSucceeds) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model)) << "Test model missing: " << model;

    OnnxSession s;
    auto result = s.load(model);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(s.is_loaded());
    EXPECT_FALSE(s.input_names().empty());
    EXPECT_FALSE(s.output_names().empty());
}

TEST(OnnxSessionTest, InputOutputNamesMatchModel) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxSession s;
    ASSERT_TRUE(s.load(model).has_value());
    EXPECT_EQ(s.input_names()[0],  "input");
    EXPECT_EQ(s.output_names()[0], "output");
}

TEST(OnnxSessionTest, RunProducesCorrectOutputShape) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxSession s;
    ASSERT_TRUE(s.load(model).has_value());

    // Input: [1, 3, 8, 8] = 192 floats
    TensorInfo input{"input", {1, 3, 8, 8}, std::vector<float>(192, 0.5f)};
    auto result = s.run({input});
    ASSERT_TRUE(result.has_value()) << result.error().message;
    ASSERT_EQ(result.value().size(), 1u);

    const auto& out = result.value()[0];
    EXPECT_EQ(out.name, "output");
    EXPECT_EQ(out.data.size(), 3u);  // 3 class scores
}

TEST(OnnxSessionTest, MoveSemantics) {
    fs::path model = kTestData / "tiny_classifier.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxSession s1;
    ASSERT_TRUE(s1.load(model).has_value());

    OnnxSession s2 = std::move(s1);
    EXPECT_TRUE(s2.is_loaded());

    TensorInfo input{"input", {1, 3, 8, 8}, std::vector<float>(192, 0.0f)};
    EXPECT_TRUE(s2.run({input}).has_value());
}
