// tests/onnx/test_onnx_segmentor.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <opencv2/core.hpp>
#include "improc/onnx/onnx_segmentor.hpp"
#include "improc/exceptions.hpp"

using namespace improc::onnx;
using improc::ml::SegmentationMask;
using improc::core::Image;
using improc::core::BGR;
namespace fs = std::filesystem;

static const fs::path kTestData =
    fs::path{__FILE__}.parent_path() / "testdata";

// ── Construction / validation ─────────────────────────────────────────────────

TEST(OnnxSegmentorTest, NonExistentPathThrows) {
    EXPECT_THROW(OnnxSegmentor{"nonexistent/model.onnx"}, improc::ModelError);
}

TEST(OnnxSegmentorTest, CorruptModelThrows) {
    fs::path p = fs::temp_directory_path() / "improc_corrupt_seg.onnx";
    { std::ofstream f(p); f << "not onnx"; }
    EXPECT_THROW(OnnxSegmentor{p}, improc::ModelError);
    fs::remove(p);
}

TEST(OnnxSegmentorTest, ZeroScaleThrows) {
    fs::path model = kTestData / "tiny_segmentor_semantic.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxSegmentor{model}.scale(0.0f), improc::ParameterError);
}

TEST(OnnxSegmentorTest, NegativeScaleThrows) {
    fs::path model = kTestData / "tiny_segmentor_semantic.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxSegmentor{model}.scale(-0.1f), improc::ParameterError);
}

TEST(OnnxSegmentorTest, ZeroInputWidthThrows) {
    fs::path model = kTestData / "tiny_segmentor_semantic.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxSegmentor{model}.input_size(0, 8), improc::ParameterError);
}

TEST(OnnxSegmentorTest, ZeroInputHeightThrows) {
    fs::path model = kTestData / "tiny_segmentor_semantic.onnx";
    ASSERT_TRUE(fs::exists(model));
    EXPECT_THROW(OnnxSegmentor{model}.input_size(8, 0), improc::ParameterError);
}

// ── Inference ─────────────────────────────────────────────────────────────────

TEST(OnnxSegmentorTest, InferenceReturnsExpected) {
    fs::path model = kTestData / "tiny_segmentor_semantic.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxSegmentor seg{model};
    seg.input_size(8, 8).swap_rb(false);

    cv::Mat mat(16, 16, CV_8UC3, cv::Scalar(128, 64, 32));
    Image<BGR> img{mat};

    auto result = seg(img);
    ASSERT_TRUE(result.has_value()) << result.error().message;
}

TEST(OnnxSegmentorTest, MaskSizeMatchesInput) {
    fs::path model = kTestData / "tiny_segmentor_semantic.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxSegmentor seg{model};
    seg.input_size(8, 8).swap_rb(false);

    cv::Mat mat(20, 30, CV_8UC3, cv::Scalar(100));
    Image<BGR> img{mat};

    auto result = seg(img);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().class_mask.rows(), 20);
    EXPECT_EQ(result.value().class_mask.cols(), 30);
}

TEST(OnnxSegmentorTest, MaskPixelsInClassRange) {
    fs::path model = kTestData / "tiny_segmentor_semantic.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxSegmentor seg{model};
    seg.input_size(8, 8).swap_rb(false);

    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(50));
    Image<BGR> img{mat};

    auto result = seg(img);
    ASSERT_TRUE(result.has_value());

    // tiny model has 4 classes → pixels must be in [0, 3]
    const cv::Mat& mask = result.value().class_mask.mat();
    for (int y = 0; y < mask.rows; ++y)
        for (int x = 0; x < mask.cols; ++x)
            EXPECT_LE(static_cast<int>(mask.at<uint8_t>(y, x)), 3);
}

TEST(OnnxSegmentorTest, LabelAtReturnsCorrectLabel) {
    fs::path model = kTestData / "tiny_segmentor_semantic.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxSegmentor seg{model};
    seg.input_size(8, 8).swap_rb(false)
       .labels({"background", "cat", "dog", "bird"});

    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(100));
    Image<BGR> img{mat};

    auto result = seg(img);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().label_at(1), "cat");
    EXPECT_EQ(result.value().label_at(3), "bird");
}

TEST(OnnxSegmentorTest, LabelAtOutOfRangeReturnsEmpty) {
    fs::path model = kTestData / "tiny_segmentor_semantic.onnx";
    ASSERT_TRUE(fs::exists(model));

    OnnxSegmentor seg{model};
    seg.input_size(8, 8).swap_rb(false).labels({"background"});

    cv::Mat mat(8, 8, CV_8UC3, cv::Scalar(100));
    Image<BGR> img{mat};

    auto result = seg(img);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result.value().label_at(99), "");
    EXPECT_EQ(result.value().label_at(-1), "");
}
