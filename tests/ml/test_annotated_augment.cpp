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

// ---- RandomFlip bbox ----

TEST(AnnotatedAugTest, RandomFlipBBoxPreservesSize) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(10.f, 20.f, 30.f, 40.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomFlip{}.p(1.0f)(std::move(ann), rng);
    EXPECT_EQ(result.image.rows(), 100);
    EXPECT_EQ(result.image.cols(), 100);
    EXPECT_EQ(result.boxes.size(), 1u);
}

TEST(AnnotatedAugTest, RandomFlipHorizontalBBoxCorrect) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(10.f, 20.f, 30.f, 40.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomFlip{}.p(1.0f).axis(Axis::Horizontal)(std::move(ann), rng);
    // new_x = W - x - w = 100 - 10 - 30 = 60
    EXPECT_FLOAT_EQ(result.boxes[0].box.x,      60.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.y,      20.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.width,  30.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.height, 40.f);
}

TEST(AnnotatedAugTest, RandomFlipVerticalBBoxCorrect) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(10.f, 20.f, 30.f, 40.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomFlip{}.p(1.0f).axis(Axis::Vertical)(std::move(ann), rng);
    // new_y = H - y - h = 100 - 20 - 40 = 40
    EXPECT_FLOAT_EQ(result.boxes[0].box.x,  10.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.y,  40.f);
}

TEST(AnnotatedAugTest, RandomFlipBBoxP0NeverFlips) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(10.f, 20.f, 30.f, 40.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomFlip{}.p(0.0f)(std::move(ann), rng);
    EXPECT_FLOAT_EQ(result.boxes[0].box.x, 10.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.y, 20.f);
}

TEST(AnnotatedAugTest, RandomFlipBBoxEmptyBoxesNoCrash) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(0));
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {}};
    std::mt19937 rng(0);
    auto result = RandomFlip{}.p(1.0f)(std::move(ann), rng);
    EXPECT_EQ(result.boxes.size(), 0u);
}

// ---- RandomResize bbox ----

TEST(AnnotatedAugTest, RandomResizeBBoxScalesProportionally) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(10.f, 20.f, 30.f, 40.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomResize{}.range(200, 200)(std::move(ann), rng);
    EXPECT_FLOAT_EQ(result.boxes[0].box.x,      20.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.y,      40.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.width,  60.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.height, 80.f);
    EXPECT_EQ(result.image.mat().rows, 200);
    EXPECT_EQ(result.image.mat().cols, 200);
}

TEST(AnnotatedAugTest, RandomResizeBBoxScalesProportionallyTallImage) {
    cv::Mat mat(100, 50, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(5.f, 10.f, 20.f, 30.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomResize{}.range(200, 200)(std::move(ann), rng);
    EXPECT_FLOAT_EQ(result.boxes[0].box.x,      20.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.y,      40.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.width,  80.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.height, 120.f);
    EXPECT_EQ(result.image.mat().rows, 400);
    EXPECT_EQ(result.image.mat().cols, 200);
}

TEST(AnnotatedAugTest, RandomResizeBBoxPreservesCount) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(10.f, 10.f, 20.f, 20.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomResize{}.range(150, 150)(std::move(ann), rng);
    EXPECT_EQ(result.boxes.size(), 1u);
}

TEST(AnnotatedAugTest, RandomResizeBBoxEmptyBoxesNoCrash) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(0));
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {}};
    std::mt19937 rng(0);
    auto result = RandomResize{}.range(100, 100)(std::move(ann), rng);
    EXPECT_EQ(result.boxes.size(), 0u);
}

// ---- RandomCrop bbox ----

TEST(AnnotatedAugTest, RandomCropBBoxImageSizeCorrect) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(10.f, 10.f, 20.f, 20.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomCrop{}.width(60).height(60)(std::move(ann), rng);
    EXPECT_EQ(result.image.rows(), 60);
    EXPECT_EQ(result.image.cols(), 60);
}

TEST(AnnotatedAugTest, RandomCropBBoxIdentityKeepsBbox) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(10.f, 10.f, 20.f, 20.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomCrop{}.width(100).height(100)(std::move(ann), rng);
    EXPECT_EQ(result.boxes.size(), 1u);
    EXPECT_FLOAT_EQ(result.boxes[0].box.x, 10.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.y, 10.f);
}

TEST(AnnotatedAugTest, RandomCropBBoxOutOfBoundsDropped) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(95.f, 10.f, 10.f, 20.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomCrop{}.width(100).height(100).min_area_ratio(0.6f)(std::move(ann), rng);
    EXPECT_EQ(result.boxes.size(), 0u);
}

TEST(AnnotatedAugTest, RandomCropBBoxPartialOverlapKept) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(95.f, 10.f, 10.f, 20.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomCrop{}.width(100).height(100).min_area_ratio(0.4f)(std::move(ann), rng);
    EXPECT_EQ(result.boxes.size(), 1u);
    EXPECT_FLOAT_EQ(result.boxes[0].box.width, 5.f);
}

TEST(AnnotatedAugTest, RandomCropBBoxEmptyBoxesNoCrash) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {}};
    std::mt19937 rng(42);
    auto result = RandomCrop{}.width(80).height(80)(std::move(ann), rng);
    EXPECT_EQ(result.boxes.size(), 0u);
}

// ---- RandomZoom bbox ----

TEST(AnnotatedAugTest, RandomZoomIdentityPreservesBbox) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(10.f, 20.f, 30.f, 40.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomZoom{}.range(1.0f, 1.0f)(std::move(ann), rng);
    EXPECT_EQ(result.image.rows(), 100);
    ASSERT_EQ(result.boxes.size(), 1u);
    EXPECT_FLOAT_EQ(result.boxes[0].box.x,      10.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.y,      20.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.width,  30.f);
    EXPECT_FLOAT_EQ(result.boxes[0].box.height, 40.f);
}

TEST(AnnotatedAugTest, RandomZoomBBoxOutputSizeUnchanged) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(30.f, 30.f, 20.f, 20.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomZoom{}.range(0.8f, 0.8f)(std::move(ann), rng);
    EXPECT_EQ(result.image.rows(), 100);
    EXPECT_EQ(result.image.cols(), 100);
}

TEST(AnnotatedAugTest, RandomZoomBBoxEmptyBoxesNoCrash) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(0));
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {}};
    std::mt19937 rng(0);
    auto result = RandomZoom{}.range(0.9f, 1.0f)(std::move(ann), rng);
    EXPECT_EQ(result.boxes.size(), 0u);
}

// ---- RandomRotate bbox ----

TEST(AnnotatedAugTest, RandomRotateBBoxPreservesImageSize) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(20.f, 20.f, 30.f, 30.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomRotate{}.range(-15.f, 15.f)(std::move(ann), rng);
    EXPECT_EQ(result.image.rows(), 100);
    EXPECT_EQ(result.image.cols(), 100);
}

TEST(AnnotatedAugTest, RandomRotateZeroDegBBoxUnchanged) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(20.f, 20.f, 30.f, 30.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomRotate{}.range(0.f, 0.f)(std::move(ann), rng);
    ASSERT_EQ(result.boxes.size(), 1u);
    EXPECT_NEAR(result.boxes[0].box.x,      20.f, 1.f);
    EXPECT_NEAR(result.boxes[0].box.y,      20.f, 1.f);
    EXPECT_NEAR(result.boxes[0].box.width,  30.f, 1.f);
    EXPECT_NEAR(result.boxes[0].box.height, 30.f, 1.f);
}

TEST(AnnotatedAugTest, RandomRotateBBoxEmptyBoxesNoCrash) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(0));
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {}};
    std::mt19937 rng(0);
    auto result = RandomRotate{}.range(-10.f, 10.f)(std::move(ann), rng);
    EXPECT_EQ(result.boxes.size(), 0u);
}

// ---- RandomShear bbox ----

TEST(AnnotatedAugTest, RandomShearBBoxPreservesImageSize) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(20.f, 20.f, 30.f, 30.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomShear{}.range(-10.f, 10.f)(std::move(ann), rng);
    EXPECT_EQ(result.image.rows(), 100);
    EXPECT_EQ(result.image.cols(), 100);
}

TEST(AnnotatedAugTest, RandomShearZeroAngleBBoxUnchanged) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(20.f, 20.f, 30.f, 30.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomShear{}.range(0.f, 0.f)(std::move(ann), rng);
    ASSERT_EQ(result.boxes.size(), 1u);
    EXPECT_NEAR(result.boxes[0].box.x,      20.f, 1.f);
    EXPECT_NEAR(result.boxes[0].box.y,      20.f, 1.f);
    EXPECT_NEAR(result.boxes[0].box.width,  30.f, 1.f);
    EXPECT_NEAR(result.boxes[0].box.height, 30.f, 1.f);
}

TEST(AnnotatedAugTest, RandomShearBBoxEmptyBoxesNoCrash) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(0));
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {}};
    std::mt19937 rng(0);
    auto result = RandomShear{}.range(-5.f, 5.f)(std::move(ann), rng);
    EXPECT_EQ(result.boxes.size(), 0u);
}

// ---- RandomPerspective bbox ----

TEST(AnnotatedAugTest, RandomPerspectiveBBoxPreservesImageSize) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(20.f, 20.f, 30.f, 30.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomPerspective{}.distortion_scale(0.3f)(std::move(ann), rng);
    EXPECT_EQ(result.image.rows(), 100);
    EXPECT_EQ(result.image.cols(), 100);
}

TEST(AnnotatedAugTest, RandomPerspectiveZeroDistortionBBoxUnchanged) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    BBox bb{cv::Rect2f(20.f, 20.f, 30.f, 30.f)};
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {bb}};
    std::mt19937 rng(42);
    auto result = RandomPerspective{}.distortion_scale(0.f)(std::move(ann), rng);
    ASSERT_EQ(result.boxes.size(), 1u);
    EXPECT_NEAR(result.boxes[0].box.x,      20.f, 1.f);
    EXPECT_NEAR(result.boxes[0].box.y,      20.f, 1.f);
    EXPECT_NEAR(result.boxes[0].box.width,  30.f, 1.f);
    EXPECT_NEAR(result.boxes[0].box.height, 30.f, 1.f);
}

TEST(AnnotatedAugTest, RandomPerspectiveBBoxEmptyBoxesNoCrash) {
    cv::Mat mat(50, 50, CV_8UC3, cv::Scalar(0));
    AnnotatedImage<BGR> ann{Image<BGR>(mat), {}};
    std::mt19937 rng(0);
    auto result = RandomPerspective{}.distortion_scale(0.5f)(std::move(ann), rng);
    EXPECT_EQ(result.boxes.size(), 0u);
}
