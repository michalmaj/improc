// tests/ml/test_dnn_detector.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include <filesystem>
#include <fstream>
#include "improc/ml/dnn_detector.hpp"

using namespace improc::ml;
namespace fs = std::filesystem;

TEST(DnnDetectorTest, NonExistentPathThrows) {
    EXPECT_THROW(DnnDetector{"nonexistent/model.onnx"}, improc::ModelError);
}

TEST(DnnDetectorTest, InvalidExtensionThrows) {
    fs::path p = fs::temp_directory_path() / "improc_det_dummy.txt";
    { std::ofstream f(p); f << "not a model"; }
    EXPECT_THROW(DnnDetector{p.string()}, improc::ModelError);
    fs::remove(p);
}

TEST(DnnDetectorTest, CorruptModelThrows) {
    fs::path p = fs::temp_directory_path() / "improc_det_dummy.onnx";
    { std::ofstream f(p); f << "not a real onnx file"; }
    EXPECT_THROW(DnnDetector{p.string()}, improc::ModelError);
    fs::remove(p);
}

TEST(DnnDetectorTest, ConfidenceThresholdBelowZeroThrows) {
    EXPECT_THROW(DnnDetector{"x.onnx"}.confidence_threshold(-0.1f), std::exception);
}

TEST(DnnDetectorTest, ConfidenceThresholdAboveOneThrows) {
    EXPECT_THROW(DnnDetector{"x.onnx"}.confidence_threshold(1.1f), std::exception);
}

TEST(DnnDetectorTest, NmsThresholdBelowZeroThrows) {
    EXPECT_THROW(DnnDetector{"x.onnx"}.nms_threshold(-0.1f), std::exception);
}

TEST(DnnDetectorTest, NmsThresholdAboveOneThrows) {
    EXPECT_THROW(DnnDetector{"x.onnx"}.nms_threshold(1.1f), std::exception);
}

TEST(DnnDetectorTest, ZeroScaleThrows) {
    EXPECT_THROW(DnnDetector{"x.onnx"}.scale(0.0f), std::exception);
}

TEST(DnnDetectorTest, ZeroInputWidthThrows) {
    EXPECT_THROW(DnnDetector{"x.onnx"}.input_size(0, 640), std::exception);
}

TEST(DnnDetectorTest, ZeroInputHeightThrows) {
    EXPECT_THROW(DnnDetector{"x.onnx"}.input_size(640, 0), std::exception);
}
