// tests/ml/test_annotated_augment.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/ml/annotated.hpp"
#include "improc/ml/augment/geometric.hpp"
#include "improc/ml/augment/bbox_compose.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;
using namespace improc::ml;

TEST(AnnotatedAugTest, BBoxDefaultClassId) {
    BBox bb{cv::Rect2f(10.f, 20.f, 30.f, 40.f)};
    EXPECT_EQ(bb.class_id, 0);
    EXPECT_EQ(bb.label, "");
}

TEST(AnnotatedAugTest, AnnotatedImageHoldsImageAndBoxes) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    Image<BGR> img(mat);
    BBox bb{cv::Rect2f(10.f, 10.f, 20.f, 20.f), 1, "cat"};
    AnnotatedImage<BGR> ann{img, {bb}};
    EXPECT_EQ(ann.image.rows(), 100);
    EXPECT_EQ(ann.boxes.size(), 1u);
    EXPECT_EQ(ann.boxes[0].class_id, 1);
    EXPECT_EQ(ann.boxes[0].label, "cat");
}

TEST(AnnotatedAugTest, OperatorPipeRoutesToOp) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(0));
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {}};
    std::mt19937 rng(0);
    auto result = std::move(ann) | RandomFlip{}.p(0.0f).bind(rng);
    EXPECT_EQ(result.image.rows(), 10);
    EXPECT_EQ(result.image.cols(), 10);
}

// ---- BBoxCompose ----

TEST(AnnotatedAugTest, BBoxComposeSequentialApplication) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(128, 0, 0));
    BBox bb{cv::Rect2f(10.f, 10.f, 30.f, 30.f), 0, "obj"};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    BBoxCompose<BGR> pipeline;
    pipeline.add([](auto a, auto& r){ return RandomFlip{}.p(0.0f)(std::move(a), r); });
    pipeline.add([](auto a, auto& r){ return RandomFlip{}.p(0.0f)(std::move(a), r); });
    auto result = pipeline(std::move(ann), rng);
    EXPECT_EQ(result.image.rows(), 100);
    EXPECT_EQ(result.boxes.size(), 1u);
}

TEST(AnnotatedAugTest, BBoxComposeBindPipeline) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(0));
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {}};
    std::mt19937 rng(0);
    BBoxCompose<BGR> pipeline;
    pipeline.add([](auto a, auto& r){ return RandomFlip{}.p(0.0f)(std::move(a), r); });
    auto result = std::move(ann) | pipeline.bind(rng);
    EXPECT_EQ(result.image.rows(), 50);
}

TEST(AnnotatedAugTest, BBoxComposeNullOpThrows) {
    BBoxCompose<BGR> pipeline;
    EXPECT_THROW(pipeline.add(nullptr), improc::ParameterError);
}

// ---- min_area_ratio setter validation ----

TEST(AnnotatedAugTest, MinAreaRatioOutOfRangeThrows) {
    EXPECT_THROW(RandomFlip{}.min_area_ratio(-0.1f),       improc::ParameterError);
    EXPECT_THROW(RandomFlip{}.min_area_ratio(1.1f),        improc::ParameterError);
    EXPECT_THROW(RandomRotate{}.min_area_ratio(-0.01f),    improc::ParameterError);
    EXPECT_THROW(RandomCrop{}.min_area_ratio(1.01f),       improc::ParameterError);
    EXPECT_THROW(RandomResize{}.min_area_ratio(-0.5f),     improc::ParameterError);
    EXPECT_THROW(RandomZoom{}.min_area_ratio(2.0f),        improc::ParameterError);
    EXPECT_THROW(RandomShear{}.min_area_ratio(-0.001f),    improc::ParameterError);
    EXPECT_THROW(RandomPerspective{}.min_area_ratio(1.1f), improc::ParameterError);
}
