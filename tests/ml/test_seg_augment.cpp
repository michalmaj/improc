// tests/ml/test_seg_augment.cpp
#include <gtest/gtest.h>
#include <random>
#include <opencv2/core.hpp>
#include "improc/ml/segmented.hpp"
#include "improc/ml/augment/geometric.hpp"

using namespace improc::ml;
using improc::core::BGR;
using improc::core::Gray;

// Helper: make a SegmentedImage with solid-colour image and given mask
static SegmentedImage<BGR> make_seg(int rows, int cols,
                                     const cv::Mat& mask_mat,
                                     bool with_instance = false) {
    cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(100, 150, 200));
    std::optional<Image<Gray>> inst;
    if (with_instance) {
        cv::Mat inst_mat(rows, cols, CV_8UC1, cv::Scalar(5));
        inst = Image<Gray>(inst_mat.clone());
    }
    return {Image<BGR>(img.clone()), Image<Gray>(mask_mat.clone()), std::move(inst)};
}

TEST(SegAugmentTest, SegmentedImageConstructs) {
    cv::Mat mask(4, 4, CV_8UC1, cv::Scalar(1));
    auto seg = make_seg(4, 4, mask);
    EXPECT_EQ(seg.image.cols(), 4);
    EXPECT_EQ(seg.class_mask.cols(), 4);
    EXPECT_FALSE(seg.instance_mask.has_value());
}
