// tests/ml/augment/test_compose.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/ml/augment/compose.hpp"
#include "improc/ml/augment/geometric.hpp"
#include "improc/ml/augment/noise.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;
using namespace improc::ml;

// ---- Compose ----

TEST(ComposeAugTest, ComposeAppliesStepsSequentially) {
    // Two flips = back to original image
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    mat.at<cv::Vec3b>(0, 0) = {255, 0, 0};
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto pipeline = Compose<BGR>{}
        .add(RandomFlip{}.p(1.0f))
        .add(RandomFlip{}.p(1.0f));
    Image<BGR> result = pipeline(img, rng);
    // two horizontal flips return to original
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 255);
}

TEST(ComposeAugTest, ComposeWithZeroStepsReturnsUnchanged) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = Compose<BGR>{}(img, rng);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 100);
}

TEST(ComposeAugTest, ComposePreservesSizeAndType) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto pipeline = Compose<BGR>{}
        .add(RandomFlip{}.p(0.5f))
        .add(RandomSaltAndPepper{}.p(0.0f));
    Image<BGR> result = pipeline(img, rng);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 10);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(ComposeAugTest, ComposeBindRngPipelineOp) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto pipeline = Compose<BGR>{}.add(RandomFlip{}.p(0.0f));
    Image<BGR> result = img | pipeline.bind(rng);
    EXPECT_EQ(result.rows(), 10);
}

// ---- RandomApply ----

TEST(ComposeAugTest, RandomApplyAtP1AlwaysApplies) {
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    mat.at<cv::Vec3b>(0, 0) = {255, 0, 0};
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto ra = RandomApply<BGR>{RandomFlip{}.p(1.0f), 1.0f};
    Image<BGR> result = ra(img, rng);
    // flip applied: marker moved to top-right
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 3)[0], 255);
}

TEST(ComposeAugTest, RandomApplyAtP0NeverApplies) {
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    mat.at<cv::Vec3b>(0, 0) = {255, 0, 0};
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto ra = RandomApply<BGR>{RandomFlip{}.p(1.0f), 0.0f};
    Image<BGR> result = ra(img, rng);
    // flip not applied: marker stays at top-left
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 255);
}

TEST(ComposeAugTest, RandomApplyInvalidPThrows) {
    EXPECT_THROW((RandomApply<BGR>{RandomFlip{}, 1.5f}), std::invalid_argument);
    EXPECT_THROW((RandomApply<BGR>{RandomFlip{}, -0.1f}), std::invalid_argument);
}

TEST(ComposeAugTest, RandomApplyBindRngPipelineOp) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto ra = RandomApply<BGR>{RandomFlip{}.p(0.0f), 0.5f};
    Image<BGR> result = img | ra.bind(rng);
    EXPECT_EQ(result.rows(), 10);
}

// ---- OneOf ----

TEST(ComposeAugTest, OneOfSingleOptionAlwaysApplies) {
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    mat.at<cv::Vec3b>(0, 0) = {255, 0, 0};
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto oo = OneOf<BGR>{}.add(RandomFlip{}.p(1.0f));
    Image<BGR> result = oo(img, rng);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 3)[0], 255);
}

TEST(ComposeAugTest, OneOfEmptyThrows) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    EXPECT_THROW(OneOf<BGR>{}(img, rng), std::logic_error);
}

TEST(ComposeAugTest, OneOfDistributesAcrossOptions) {
    // Two options: p=1 flip vs p=0 noop. Over many calls, both should appear.
    cv::Mat mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));
    mat.at<cv::Vec3b>(0, 0) = {255, 0, 0};
    Image<BGR> img(mat);
    std::mt19937 rng(0);
    auto oo = OneOf<BGR>{}
        .add(RandomFlip{}.p(1.0f))               // option 0: always flips → marker at (0,3)
        .add(RandomSaltAndPepper{}.p(0.0f));      // option 1: noop → marker at (0,0)
    int flipped = 0, unchanged = 0;
    for (int i = 0; i < 100; ++i) {
        Image<BGR> result = oo(img, rng);
        if (result.mat().at<cv::Vec3b>(0, 3)[0] == 255) ++flipped;
        else if (result.mat().at<cv::Vec3b>(0, 0)[0] == 255) ++unchanged;
    }
    EXPECT_GT(flipped,    0);
    EXPECT_GT(unchanged,  0);
}

TEST(ComposeAugTest, OneOfBindRngPipelineOp) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto oo = OneOf<BGR>{}.add(RandomFlip{}.p(0.0f));
    Image<BGR> result = img | oo.bind(rng);
    EXPECT_EQ(result.rows(), 10);
}

TEST(ComposeAugTest, NestedComposeInsideCompose) {
    cv::Mat mat(10, 10, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto inner = Compose<BGR>{}.add(RandomFlip{}.p(0.0f));
    auto outer = Compose<BGR>{}.add(inner).add(RandomSaltAndPepper{}.p(0.0f));
    Image<BGR> result = outer(img, rng);
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.mat().at<cv::Vec3b>(0, 0)[0], 100);  // unchanged with p=0
}
