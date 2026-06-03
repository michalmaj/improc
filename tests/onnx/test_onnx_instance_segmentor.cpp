// tests/onnx/test_onnx_instance_segmentor.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include "improc/onnx/onnx_instance_segmentor.hpp"
#include "improc/exceptions.hpp"

using namespace improc::onnx;
using improc::ml::SegmentInstance;
using improc::core::Image;
using improc::core::BGR;
namespace fs = std::filesystem;

static const fs::path kTestData =
    fs::path{__FILE__}.parent_path() / "testdata";

// ── Construction / validation ─────────────────────────────────────────────────

TEST(OnnxInstanceSegmentorTest, NonExistentPathThrows) {
    EXPECT_THROW(OnnxInstanceSegmentor{"nonexistent/model.onnx"}, improc::ModelError);
}

TEST(OnnxInstanceSegmentorTest, CorruptModelThrows) {
    fs::path p = fs::temp_directory_path() / "improc_corrupt_inst_seg.onnx";
    { std::ofstream f(p); f << "not onnx"; }
    EXPECT_THROW(OnnxInstanceSegmentor{p}, improc::ModelError);
    fs::remove(p);
}

TEST(OnnxInstanceSegmentorTest, NegativeConfidenceThresholdThrows) {
    fs::path model = kTestData / "tiny_instance_segmentor.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxInstanceSegmentor{model}.confidence_threshold(-0.1f),
                 improc::ParameterError);
}

TEST(OnnxInstanceSegmentorTest, ConfidenceThresholdAboveOneThrows) {
    fs::path model = kTestData / "tiny_instance_segmentor.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxInstanceSegmentor{model}.confidence_threshold(1.1f),
                 improc::ParameterError);
}

TEST(OnnxInstanceSegmentorTest, NegativeNmsThresholdThrows) {
    fs::path model = kTestData / "tiny_instance_segmentor.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxInstanceSegmentor{model}.nms_threshold(-0.1f),
                 improc::ParameterError);
}

TEST(OnnxInstanceSegmentorTest, NegativeMaskThresholdThrows) {
    fs::path model = kTestData / "tiny_instance_segmentor.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxInstanceSegmentor{model}.mask_threshold(-0.1f),
                 improc::ParameterError);
}

TEST(OnnxInstanceSegmentorTest, ZeroScaleThrows) {
    fs::path model = kTestData / "tiny_instance_segmentor.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxInstanceSegmentor{model}.scale(0.0f), improc::ParameterError);
}

TEST(OnnxInstanceSegmentorTest, ZeroInputSizeThrows) {
    fs::path model = kTestData / "tiny_instance_segmentor.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxInstanceSegmentor{model}.input_size(0, 8), improc::ParameterError);
}

// ── Inference ─────────────────────────────────────────────────────────────────

TEST(OnnxInstanceSegmentorTest, InferenceReturnsExpected) {
    fs::path model = kTestData / "tiny_instance_segmentor.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxInstanceSegmentor seg{model};
    seg.input_size(8, 8).swap_rb(false);

    cv::Mat mat(16, 16, CV_8UC3, cv::Scalar(128, 64, 32));
    Image<BGR> img{mat};

    auto result = seg(img);
    ASSERT_TRUE(result.has_value()) << result.error().message;
}

TEST(OnnxInstanceSegmentorTest, LowThresholdYieldsInstances) {
    fs::path model = kTestData / "tiny_instance_segmentor.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxInstanceSegmentor seg{model};
    seg.input_size(8, 8)
       .confidence_threshold(0.0f)  // keep everything
       .nms_threshold(1.0f)          // disable NMS
       .swap_rb(false);

    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(200));
    Image<BGR> img{mat};

    auto result = seg(img);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_GT(result.value().size(), 0u);
}

TEST(OnnxInstanceSegmentorTest, MaskSizeMatchesInput) {
    fs::path model = kTestData / "tiny_instance_segmentor.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxInstanceSegmentor seg{model};
    seg.input_size(8, 8)
       .confidence_threshold(0.0f)
       .nms_threshold(1.0f)
       .swap_rb(false);

    cv::Mat mat(20, 30, CV_8UC3, cv::Scalar(100));
    Image<BGR> img{mat};

    auto result = seg(img);
    ASSERT_TRUE(result.has_value());
    for (const auto& inst : result.value()) {
        EXPECT_EQ(inst.mask.rows(), 20);
        EXPECT_EQ(inst.mask.cols(), 30);
    }
}

TEST(OnnxInstanceSegmentorTest, MaskIsBinary) {
    fs::path model = kTestData / "tiny_instance_segmentor.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxInstanceSegmentor seg{model};
    seg.input_size(8, 8)
       .confidence_threshold(0.0f)
       .nms_threshold(1.0f)
       .swap_rb(false);

    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(150));
    Image<BGR> img{mat};

    auto result = seg(img);
    ASSERT_TRUE(result.has_value());
    for (const auto& inst : result.value()) {
        const cv::Mat& m = inst.mask.mat();
        for (int y = 0; y < m.rows; ++y)
            for (int x = 0; x < m.cols; ++x) {
                uint8_t px = m.at<uint8_t>(y, x);
                EXPECT_TRUE(px == 0 || px == 255)
                    << "Non-binary pixel " << static_cast<int>(px)
                    << " at (" << y << "," << x << ")";
            }
    }
}

TEST(OnnxInstanceSegmentorTest, LabelsPopulatedForKnownClassIds) {
    fs::path model = kTestData / "tiny_instance_segmentor.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxInstanceSegmentor seg{model};
    seg.input_size(8, 8)
       .confidence_threshold(0.0f)
       .nms_threshold(1.0f)
       .labels({"cat", "dog"})
       .swap_rb(false);

    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(100));
    Image<BGR> img{mat};

    auto result = seg(img);
    ASSERT_TRUE(result.has_value());
    for (const auto& inst : result.value()) {
        if (inst.class_id < 2)
            EXPECT_FALSE(inst.label.empty());
    }
}
