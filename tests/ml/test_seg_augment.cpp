// tests/ml/test_seg_augment.cpp
#include <gtest/gtest.h>
#include <random>
#include <opencv2/core.hpp>
#include "improc/ml/segmented.hpp"
#include "improc/ml/augment/geometric.hpp"

using namespace improc::ml;
using improc::core::BGR;
using improc::core::Gray;
using improc::ParameterError;

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

TEST(SegAugmentTest, SegmentedImageWithInstanceMask) {
    cv::Mat mask(4, 4, CV_8UC1, cv::Scalar(1));
    auto seg = make_seg(4, 4, mask, /*with_instance=*/true);
    ASSERT_TRUE(seg.instance_mask.has_value());
    EXPECT_EQ(seg.instance_mask->cols(), 4);
    EXPECT_EQ(seg.instance_mask->rows(), 4);
}

// Helper: quadrant mask — top-left=0, top-right=1, bottom-left=2, bottom-right=3
static cv::Mat make_quadrant_mask(int rows, int cols) {
    cv::Mat m(rows, cols, CV_8UC1);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            m.at<uint8_t>(r, c) = static_cast<uint8_t>((r < rows/2 ? 0 : 2) + (c < cols/2 ? 0 : 1));
    return m;
}

// Helper: check all pixels in mat are in allowed_values
static bool only_values(const cv::Mat& m, std::initializer_list<uint8_t> allowed) {
    for (int r = 0; r < m.rows; ++r)
        for (int c = 0; c < m.cols; ++c) {
            uint8_t v = m.at<uint8_t>(r, c);
            bool found = false;
            for (auto a : allowed) found |= (v == a);
            if (!found) return false;
        }
    return true;
}

// --- RandomFlip ---
TEST(SegAugmentTest, RandomFlipMaskSizePreserved) {
    std::mt19937 rng{0};
    auto seg = make_seg(8, 8, make_quadrant_mask(8, 8));
    auto out = RandomFlip{}.p(1.0f)(seg, rng);
    EXPECT_EQ(out.class_mask.rows(), seg.image.rows());
    EXPECT_EQ(out.class_mask.cols(), seg.image.cols());
}

TEST(SegAugmentTest, RandomFlipMaskValuesCorrect) {
    std::mt19937 rng{0};
    cv::Mat mask = make_quadrant_mask(4, 4);
    auto seg = make_seg(4, 4, mask);
    // p=1 horizontal flip: left↔right, so 0↔1 and 2↔3
    auto out = RandomFlip{}.p(1.0f)(seg, rng);
    // top-left should now be 1 (was 0 before flip)
    EXPECT_EQ(out.class_mask.mat().at<uint8_t>(0, 0), 1);
    EXPECT_EQ(out.class_mask.mat().at<uint8_t>(0, 2), 0);
    EXPECT_EQ(out.class_mask.mat().at<uint8_t>(2, 0), 3);
    EXPECT_EQ(out.class_mask.mat().at<uint8_t>(2, 2), 2);
}

TEST(SegAugmentTest, RandomFlipInstanceMaskPropagated) {
    std::mt19937 rng{0};
    auto seg = make_seg(8, 8, make_quadrant_mask(8, 8), /*with_instance=*/true);
    auto out = RandomFlip{}.p(1.0f)(seg, rng);
    ASSERT_TRUE(out.instance_mask.has_value());
    EXPECT_EQ(out.instance_mask->rows(), out.image.rows());
    EXPECT_EQ(out.instance_mask->cols(), out.image.cols());
}

// --- RandomRotate ---
TEST(SegAugmentTest, RandomRotateMaskSameSize) {
    std::mt19937 rng{42};
    cv::Mat mask(64, 64, CV_8UC1);
    cv::randu(mask, 0, 4);
    auto seg = make_seg(64, 64, mask);
    auto out = RandomRotate{}.range(45.0f, 45.0f)(seg, rng);
    EXPECT_EQ(out.class_mask.rows(), out.image.rows());
    EXPECT_EQ(out.class_mask.cols(), out.image.cols());
}

TEST(SegAugmentTest, RandomRotateMaskInterNearest) {
    std::mt19937 rng{42};
    cv::Mat mask(64, 64, CV_8UC1, cv::Scalar(0));
    mask(cv::Rect(32, 32, 32, 32)).setTo(1);
    auto seg = make_seg(64, 64, mask);
    auto out = RandomRotate{}.range(30.0f, 30.0f)(seg, rng);
    EXPECT_TRUE(only_values(out.class_mask.mat(), {0, 1}));
}

// --- RandomCrop ---
TEST(SegAugmentTest, RandomCropMaskCroppedToSameROI) {
    std::mt19937 rng{0};
    auto seg = make_seg(64, 64, make_quadrant_mask(64, 64));
    auto out = RandomCrop{}.width(32).height(32)(seg, rng);
    EXPECT_EQ(out.image.cols(), 32);
    EXPECT_EQ(out.image.rows(), 32);
    EXPECT_EQ(out.class_mask.cols(), 32);
    EXPECT_EQ(out.class_mask.rows(), 32);
}

// --- RandomResize ---
TEST(SegAugmentTest, RandomResizeMaskSameAsImage) {
    std::mt19937 rng{0};
    cv::Mat mask(64, 64, CV_8UC1, cv::Scalar(2));
    auto seg = make_seg(64, 64, mask);
    auto out = RandomResize{}.range(128, 128)(seg, rng);
    EXPECT_EQ(out.class_mask.rows(), out.image.rows());
    EXPECT_EQ(out.class_mask.cols(), out.image.cols());
}

TEST(SegAugmentTest, RandomResizeMaskInterNearest) {
    std::mt19937 rng{0};
    cv::Mat mask(32, 32, CV_8UC1, cv::Scalar(0));
    mask(cv::Rect(16, 0, 16, 32)).setTo(1);
    auto seg = make_seg(32, 32, mask);
    auto out = RandomResize{}.range(64, 64)(seg, rng);
    EXPECT_TRUE(only_values(out.class_mask.mat(), {0, 1}));
}

// --- RandomZoom ---
TEST(SegAugmentTest, RandomZoomMaskSameAsImage) {
    std::mt19937 rng{0};
    auto seg = make_seg(64, 64, make_quadrant_mask(64, 64));
    auto out = RandomZoom{}.range(0.5f, 0.5f)(seg, rng);
    EXPECT_EQ(out.class_mask.rows(), out.image.rows());
    EXPECT_EQ(out.class_mask.cols(), out.image.cols());
}

TEST(SegAugmentTest, RandomZoomMaskInterNearest) {
    std::mt19937 rng{0};
    cv::Mat mask(32, 32, CV_8UC1, cv::Scalar(0));
    mask(cv::Rect(16, 0, 16, 32)).setTo(1);
    auto seg = make_seg(32, 32, mask);
    auto out = RandomZoom{}.range(0.5f, 0.5f)(seg, rng);
    EXPECT_TRUE(only_values(out.class_mask.mat(), {0, 1}));
}

// --- RandomShear ---
TEST(SegAugmentTest, RandomShearMaskSameAsImage) {
    std::mt19937 rng{0};
    auto seg = make_seg(64, 64, make_quadrant_mask(64, 64));
    auto out = RandomShear{}.range(10.0f, 10.0f)(seg, rng);
    EXPECT_EQ(out.class_mask.rows(), out.image.rows());
    EXPECT_EQ(out.class_mask.cols(), out.image.cols());
}

TEST(SegAugmentTest, RandomShearMaskInterNearest) {
    std::mt19937 rng{0};
    cv::Mat mask(64, 64, CV_8UC1, cv::Scalar(0));
    mask(cv::Rect(32, 0, 32, 64)).setTo(1);
    auto seg = make_seg(64, 64, mask);
    auto out = RandomShear{}.range(10.0f, 10.0f)(seg, rng);
    EXPECT_TRUE(only_values(out.class_mask.mat(), {0, 1}));
}

// --- RandomPerspective ---
TEST(SegAugmentTest, RandomPerspectiveMaskSameAsImage) {
    std::mt19937 rng{0};
    auto seg = make_seg(64, 64, make_quadrant_mask(64, 64));
    auto out = RandomPerspective{}.distortion_scale(0.3f)(seg, rng);
    EXPECT_EQ(out.class_mask.rows(), out.image.rows());
    EXPECT_EQ(out.class_mask.cols(), out.image.cols());
}

TEST(SegAugmentTest, RandomPerspectiveMaskInterNearest) {
    std::mt19937 rng{0};
    cv::Mat mask(64, 64, CV_8UC1, cv::Scalar(0));
    mask(cv::Rect(32, 0, 32, 64)).setTo(1);
    auto seg = make_seg(64, 64, mask);
    auto out = RandomPerspective{}.distortion_scale(0.2f)(seg, rng);
    EXPECT_TRUE(only_values(out.class_mask.mat(), {0, 1}));
}

// instance mask nullopt survives transform
TEST(SegAugmentTest, NulloptInstanceMaskRemains) {
    std::mt19937 rng{0};
    auto seg = make_seg(8, 8, make_quadrant_mask(8, 8), /*with_instance=*/false);
    ASSERT_FALSE(seg.instance_mask.has_value());
    auto out = RandomFlip{}.p(1.0f)(seg, rng);
    EXPECT_FALSE(out.instance_mask.has_value());
}

TEST(SegAugmentTest, RandomRotateInstanceMaskPropagated) {
    std::mt19937 rng{42};
    auto seg = make_seg(64, 64, make_quadrant_mask(64, 64), /*with_instance=*/true);
    auto out = RandomRotate{}.range(30.0f, 30.0f)(seg, rng);
    ASSERT_TRUE(out.instance_mask.has_value());
    EXPECT_EQ(out.instance_mask->rows(), out.image.rows());
    EXPECT_EQ(out.instance_mask->cols(), out.image.cols());
}

TEST(SegAugmentTest, RandomCropInstanceMaskPropagated) {
    std::mt19937 rng{0};
    auto seg = make_seg(64, 64, make_quadrant_mask(64, 64), /*with_instance=*/true);
    auto out = RandomCrop{}.width(32).height(32)(seg, rng);
    ASSERT_TRUE(out.instance_mask.has_value());
    EXPECT_EQ(out.instance_mask->rows(), 32);
    EXPECT_EQ(out.instance_mask->cols(), 32);
}

TEST(SegAugmentTest, RandomResizeInstanceMaskPropagated) {
    std::mt19937 rng{0};
    cv::Mat mask(64, 64, CV_8UC1, cv::Scalar(2));
    auto seg = make_seg(64, 64, mask, /*with_instance=*/true);
    auto out = RandomResize{}.range(128, 128)(seg, rng);
    ASSERT_TRUE(out.instance_mask.has_value());
    EXPECT_EQ(out.instance_mask->rows(), out.image.rows());
    EXPECT_EQ(out.instance_mask->cols(), out.image.cols());
}

#include "improc/ml/augment/color.hpp"
#include "improc/ml/augment/noise.hpp"
#include "improc/ml/augment/blur.hpp"
#include "improc/ml/augment/seg_compose.hpp"

// Helper: compare two Gray mats pixel-by-pixel
static bool mats_equal(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.type() != b.type()) return false;
    return cv::countNonZero(a != b) == 0;
}

TEST(SegAugmentTest, RandomBrightnessMaskUnchanged) {
    std::mt19937 rng{0};
    cv::Mat mask = make_quadrant_mask(32, 32);
    auto seg = make_seg(32, 32, mask);
    auto out = RandomBrightness{}(seg, rng);
    EXPECT_TRUE(mats_equal(out.class_mask.mat(), mask));
}

TEST(SegAugmentTest, RandomContrastMaskUnchanged) {
    std::mt19937 rng{0};
    cv::Mat mask = make_quadrant_mask(32, 32);
    auto seg = make_seg(32, 32, mask);
    auto out = RandomContrast{}(seg, rng);
    EXPECT_TRUE(mats_equal(out.class_mask.mat(), mask));
}

TEST(SegAugmentTest, ColorJitterMaskUnchanged) {
    std::mt19937 rng{0};
    cv::Mat mask = make_quadrant_mask(32, 32);
    auto seg = make_seg(32, 32, mask);
    auto out = ColorJitter{}(seg, rng);
    EXPECT_TRUE(mats_equal(out.class_mask.mat(), mask));
}

TEST(SegAugmentTest, RandomGaussianNoiseMaskUnchanged) {
    std::mt19937 rng{0};
    cv::Mat mask = make_quadrant_mask(32, 32);
    auto seg = make_seg(32, 32, mask);
    auto out = RandomGaussianNoise{}(seg, rng);
    EXPECT_TRUE(mats_equal(out.class_mask.mat(), mask));
}

TEST(SegAugmentTest, RandomSaltAndPepperMaskUnchanged) {
    std::mt19937 rng{0};
    cv::Mat mask = make_quadrant_mask(32, 32);
    auto seg = make_seg(32, 32, mask);
    auto out = RandomSaltAndPepper{}(seg, rng);
    EXPECT_TRUE(mats_equal(out.class_mask.mat(), mask));
}

TEST(SegAugmentTest, RandomBlurMaskUnchanged) {
    std::mt19937 rng{0};
    cv::Mat mask = make_quadrant_mask(32, 32);
    auto seg = make_seg(32, 32, mask);
    auto out = RandomBlur{}(seg, rng);
    EXPECT_TRUE(mats_equal(out.class_mask.mat(), mask));
}

TEST(SegAugmentTest, SegComposeSingleOp) {
    std::mt19937 rng{0};
    cv::Mat mask = make_quadrant_mask(32, 32);
    auto seg = make_seg(32, 32, mask);
    SegCompose<BGR> comp;
    comp.add([](SegmentedImage<BGR> s, std::mt19937& r) {
        return RandomBrightness{}(std::move(s), r);
    });
    auto out = comp(seg, rng);
    EXPECT_EQ(out.class_mask.rows(), 32);
    EXPECT_TRUE(mats_equal(out.class_mask.mat(), mask));
}

TEST(SegAugmentTest, SegComposeTwoOps) {
    std::mt19937 rng{0};
    cv::Mat mask = make_quadrant_mask(32, 32);
    auto seg = make_seg(32, 32, mask);
    SegCompose<BGR> comp;
    comp.add([](SegmentedImage<BGR> s, std::mt19937& r) { return RandomFlip{}.p(1.0f)(std::move(s), r); });
    comp.add([](SegmentedImage<BGR> s, std::mt19937& r) { return RandomBrightness{}(std::move(s), r); });
    auto out = comp(seg, rng);
    // After horizontal flip: top-left of mask is 1 (was 0)
    EXPECT_EQ(out.class_mask.mat().at<uint8_t>(0, 0), 1);
}

TEST(SegAugmentTest, SegComposeNullOpThrows) {
    SegCompose<BGR> comp;
    EXPECT_THROW(comp.add(nullptr), ParameterError);
}

TEST(SegAugmentTest, SegPipelineOperator) {
    std::mt19937 rng{0};
    cv::Mat mask = make_quadrant_mask(32, 32);
    auto seg = make_seg(32, 32, mask);
    auto out = seg | RandomFlip{}.p(1.0f).bind(rng);
    EXPECT_EQ(out.class_mask.mat().at<uint8_t>(0, 0), 1);
}

TEST(SegAugmentTest, ColorOpInstanceMaskUnchanged) {
    std::mt19937 rng{0};
    cv::Mat mask = make_quadrant_mask(32, 32);
    cv::Mat inst_mat(32, 32, CV_8UC1, cv::Scalar(7));
    auto seg = make_seg(32, 32, mask, /*with_instance=*/true);
    auto out = RandomBrightness{}(seg, rng);
    ASSERT_TRUE(out.instance_mask.has_value());
    EXPECT_TRUE(mats_equal(out.instance_mask->mat(), seg.instance_mask->mat()));
}
