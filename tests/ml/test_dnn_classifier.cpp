// tests/ml/test_dnn_classifier.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include <filesystem>
#include <fstream>
#include "improc/ml/dnn_classifier.hpp"

using namespace improc::ml;
namespace fs = std::filesystem;

TEST(DnnClassifierTest, NonExistentPathThrows) {
    EXPECT_THROW(DnnClassifier{"nonexistent/model.onnx"}, improc::ModelError);
}

TEST(DnnClassifierTest, InvalidExtensionThrows) {
    fs::path p = fs::temp_directory_path() / "improc_test_dummy.txt";
    { std::ofstream f(p); f << "not a model"; }
    EXPECT_THROW(DnnClassifier{p.string()}, improc::ModelError);
    fs::remove(p);
}

TEST(DnnClassifierTest, CorruptModelThrows) {
    fs::path p = fs::temp_directory_path() / "improc_test_dummy.onnx";
    { std::ofstream f(p); f << "not a real onnx file"; }
    EXPECT_THROW(DnnClassifier{p.string()}, improc::ModelError);
    fs::remove(p);
}

TEST(DnnClassifierTest, ZeroTopKThrows) {
    EXPECT_THROW(DnnClassifier{"x.onnx"}.top_k(0), std::exception);
}

TEST(DnnClassifierTest, NegativeTopKThrows) {
    EXPECT_THROW(DnnClassifier{"x.onnx"}.top_k(-1), std::exception);
}

TEST(DnnClassifierTest, ZeroScaleThrows) {
    EXPECT_THROW(DnnClassifier{"x.onnx"}.scale(0.0f), std::exception);
}

TEST(DnnClassifierTest, NegativeScaleThrows) {
    EXPECT_THROW(DnnClassifier{"x.onnx"}.scale(-1.0f), std::exception);
}

TEST(DnnClassifierTest, ZeroInputWidthThrows) {
    EXPECT_THROW(DnnClassifier{"x.onnx"}.input_size(0, 224), std::exception);
}

TEST(DnnClassifierTest, ZeroInputHeightThrows) {
    EXPECT_THROW(DnnClassifier{"x.onnx"}.input_size(224, 0), std::exception);
}
