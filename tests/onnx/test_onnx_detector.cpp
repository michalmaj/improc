// tests/onnx/test_onnx_detector.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include "improc/onnx/onnx_detector.hpp"
#include "improc/exceptions.hpp"

using namespace improc::onnx;
using improc::core::Image;
using improc::core::BGR;
namespace fs = std::filesystem;

static const fs::path kTestData =
    fs::path{__FILE__}.parent_path() / "testdata";

// ── Construction / validation ─────────────────────────────────────────────────

TEST(OnnxDetectorTest, NonExistentPathThrows) {
    EXPECT_THROW(OnnxDetector{"nonexistent/model.onnx"}, improc::ModelError);
}

TEST(OnnxDetectorTest, CorruptModelThrows) {
    fs::path p = fs::temp_directory_path() / "improc_corrupt_det.onnx";
    { std::ofstream f(p); f << "not onnx"; }
    EXPECT_THROW(OnnxDetector{p}, improc::ModelError);
    fs::remove(p);
}

TEST(OnnxDetectorTest, NegativeConfidenceThresholdThrows) {
    fs::path model = kTestData / "tiny_detector_yolov8.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxDetector{model}.confidence_threshold(-0.1f), improc::ParameterError);
}

TEST(OnnxDetectorTest, ConfidenceThresholdAboveOneThrows) {
    fs::path model = kTestData / "tiny_detector_yolov8.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxDetector{model}.confidence_threshold(1.1f), improc::ParameterError);
}

TEST(OnnxDetectorTest, NegativeNmsThresholdThrows) {
    fs::path model = kTestData / "tiny_detector_yolov8.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxDetector{model}.nms_threshold(-0.1f), improc::ParameterError);
}

TEST(OnnxDetectorTest, ZeroInputWidthThrows) {
    fs::path model = kTestData / "tiny_detector_yolov8.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxDetector{model}.input_size(0, 8), improc::ParameterError);
}

TEST(OnnxDetectorTest, ZeroScaleThrows) {
    fs::path model = kTestData / "tiny_detector_yolov8.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxDetector{model}.scale(0.0f), improc::ParameterError);
}

// ── Inference (YOLOv8 tiny model) ─────────────────────────────────────────────

TEST(OnnxDetectorTest, YoloV8InferenceReturnsExpected) {
    fs::path model = kTestData / "tiny_detector_yolov8.onnx";
    ASSERT_TRUE(fs::exists(model));

    // tiny_detector_yolov8 expects 8×8 input; output [1, 6, 10]
    // With random weights most boxes won't pass the 0.5 threshold → empty is fine
    OnnxDetector det{model};
    det.input_size(8, 8).confidence_threshold(0.5f).swap_rb(false);

    cv::Mat mat(16, 16, CV_8UC3, cv::Scalar(128, 64, 32));
    Image<BGR> img{mat};

    auto result = det(img);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    // No assertion on count — random model may or may not produce detections
}

TEST(OnnxDetectorTest, YoloV8AllDetectionsWithLowThreshold) {
    fs::path model = kTestData / "tiny_detector_yolov8.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxDetector det{model};
    det.input_size(8, 8)
       .confidence_threshold(0.0f)  // keep everything
       .nms_threshold(1.0f)         // disable NMS
       .swap_rb(false);

    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(200));
    Image<BGR> img{mat};

    auto result = det(img);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    // At threshold 0.0 we should see at least some boxes from 10-box output
    EXPECT_GT(result.value().size(), 0u);
}

TEST(OnnxDetectorTest, LabelsArePopulatedWhenProvided) {
    fs::path model = kTestData / "tiny_detector_yolov8.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxDetector det{model};
    det.input_size(8, 8)
       .confidence_threshold(0.0f)
       .nms_threshold(1.0f)
       .labels({"classA", "classB"})
       .swap_rb(false);

    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(100));
    Image<BGR> img{mat};

    auto result = det(img);
    ASSERT_TRUE(result.has_value());
    for (const auto& d : result.value()) {
        if (d.class_id < 2)
            EXPECT_FALSE(d.label.empty());
    }
}
