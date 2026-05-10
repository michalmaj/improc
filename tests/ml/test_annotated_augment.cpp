// tests/ml/test_annotated_augment.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/ml/annotated.hpp"
#include "improc/ml/augment/geometric.hpp"

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
