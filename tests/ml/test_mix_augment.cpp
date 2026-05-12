// tests/ml/test_mix_augment.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/ml/labeled.hpp"
#include "improc/ml/augment/mixup.hpp"
#include "improc/exceptions.hpp"

using namespace improc;
using namespace improc::core;
using namespace improc::ml;

// ---- LabeledImage ----

TEST(MixAugTest, LabeledImageHoldsImageAndLabel) {
    cv::Mat mat(100, 100, CV_8UC3, cv::Scalar(0));
    LabeledImage<BGR> li{Image<BGR>(mat), {1.0f, 0.0f, 0.0f}};
    EXPECT_EQ(li.image.rows(), 100);
    EXPECT_EQ(li.image.cols(), 100);
    ASSERT_EQ(li.label.size(), 3u);
    EXPECT_FLOAT_EQ(li.label[0], 1.0f);
    EXPECT_FLOAT_EQ(li.label[1], 0.0f);
}

// ---- MixUp ----

TEST(MixAugTest, MixUpPZeroReturnsIdentical) {
    cv::Mat m1(100, 100, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::Mat m2(100, 100, CV_8UC3, cv::Scalar(200, 200, 200));
    LabeledImage<BGR> a{Image<BGR>(m1), {1.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m2), {0.0f, 1.0f}};
    std::mt19937 rng(42);
    auto result = MixUp{}.p(0.0f)(a, b, rng);
    EXPECT_EQ(cv::norm(result.image.mat(), m1, cv::NORM_INF), 0.0);
    EXPECT_EQ(result.label, a.label);
}

TEST(MixAugTest, MixUpPOneMixesImages) {
    cv::Mat m1(100, 100, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::Mat m2(100, 100, CV_8UC3, cv::Scalar(200, 200, 200));
    LabeledImage<BGR> a{Image<BGR>(m1), {1.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m2), {0.0f, 1.0f}};
    std::mt19937 rng(42);
    auto result = MixUp{}.alpha(0.4f).p(1.0f)(a, b, rng);
    EXPECT_GT(cv::norm(result.image.mat(), m1, cv::NORM_INF), 0.0);
    EXPECT_GT(cv::norm(result.image.mat(), m2, cv::NORM_INF), 0.0);
}

TEST(MixAugTest, MixUpOutputSizeMatchesInput) {
    cv::Mat m1(80, 120, CV_8UC3, cv::Scalar(0));
    cv::Mat m2(80, 120, CV_8UC3, cv::Scalar(128));
    LabeledImage<BGR> a{Image<BGR>(m1), {1.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m2), {0.0f, 1.0f}};
    std::mt19937 rng(42);
    auto result = MixUp{}.p(1.0f)(a, b, rng);
    EXPECT_EQ(result.image.rows(), 80);
    EXPECT_EQ(result.image.cols(), 120);
    EXPECT_EQ(result.image.mat().type(), CV_8UC3);
}

TEST(MixAugTest, MixUpBlendsSoftLabels) {
    cv::Mat m1(10, 10, CV_8UC3, cv::Scalar(0));
    cv::Mat m2(10, 10, CV_8UC3, cv::Scalar(255));
    LabeledImage<BGR> a{Image<BGR>(m1), {1.0f, 0.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m2), {0.0f, 0.0f, 1.0f}};
    std::mt19937 rng(0);
    auto result = MixUp{}.alpha(0.4f).p(1.0f)(a, b, rng);
    ASSERT_EQ(result.label.size(), 3u);
    for (float v : result.label) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
    }
    float sum = result.label[0] + result.label[1] + result.label[2];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST(MixAugTest, MixUpDeterministicSeed) {
    cv::Mat m1(50, 50, CV_8UC3, cv::Scalar(50));
    cv::Mat m2(50, 50, CV_8UC3, cv::Scalar(200));
    LabeledImage<BGR> a{Image<BGR>(m1), {1.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m2), {0.0f, 1.0f}};
    std::mt19937 rng1(99), rng2(99);
    auto r1 = MixUp{}.alpha(0.4f)(a, b, rng1);
    auto r2 = MixUp{}.alpha(0.4f)(a, b, rng2);
    EXPECT_EQ(cv::norm(r1.image.mat(), r2.image.mat(), cv::NORM_INF), 0.0);
    EXPECT_EQ(r1.label, r2.label);
}

TEST(MixAugTest, MixUpThrowsOnSizeMismatch) {
    cv::Mat m1(100, 100, CV_8UC3, cv::Scalar(0));
    cv::Mat m2(200, 200, CV_8UC3, cv::Scalar(0));
    LabeledImage<BGR> a{Image<BGR>(m1), {1.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m2), {0.0f, 1.0f}};
    std::mt19937 rng(0);
    EXPECT_THROW(MixUp{}.p(1.0f)(a, b, rng), ParameterError);
}

TEST(MixAugTest, MixUpThrowsOnLabelSizeMismatch) {
    cv::Mat m(50, 50, CV_8UC3, cv::Scalar(0));
    LabeledImage<BGR> a{Image<BGR>(m), {1.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m.clone()), {0.0f, 0.5f, 0.5f}};
    std::mt19937 rng(0);
    EXPECT_THROW(MixUp{}(a, b, rng), ParameterError);
}

TEST(MixAugTest, MixUpThrowsOnEmptyLabel) {
    cv::Mat m(50, 50, CV_8UC3, cv::Scalar(0));
    LabeledImage<BGR> a{Image<BGR>(m), {}};
    LabeledImage<BGR> b{Image<BGR>(m.clone()), {1.0f}};
    std::mt19937 rng(0);
    EXPECT_THROW(MixUp{}(a, b, rng), ParameterError);
}

// ---- CutMix ----

TEST(MixAugTest, CutMixPZeroReturnsIdentical) {
    cv::Mat m1(100, 100, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::Mat m2(100, 100, CV_8UC3, cv::Scalar(200, 200, 200));
    LabeledImage<BGR> a{Image<BGR>(m1), {1.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m2), {0.0f, 1.0f}};
    std::mt19937 rng(42);
    auto result = CutMix{}.p(0.0f)(a, b, rng);
    EXPECT_EQ(cv::norm(result.image.mat(), m1, cv::NORM_INF), 0.0);
    EXPECT_EQ(result.label, a.label);
}

TEST(MixAugTest, CutMixPOnePastesRegion) {
    cv::Mat m1(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat m2(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    LabeledImage<BGR> a{Image<BGR>(m1), {1.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m2), {0.0f, 1.0f}};
    std::mt19937 rng(42);
    auto result = CutMix{}.alpha(1.0f).p(1.0f)(a, b, rng);
    EXPECT_GT(cv::norm(result.image.mat(), m1, cv::NORM_INF), 0.0);
    ASSERT_EQ(result.label.size(), 2u);
    for (float v : result.label) {
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
    }
    EXPECT_NEAR(result.label[0] + result.label[1], 1.0f, 1e-5f);
}

TEST(MixAugTest, CutMixOutputSizeMatchesInput) {
    cv::Mat m1(80, 120, CV_8UC3, cv::Scalar(0));
    cv::Mat m2(80, 120, CV_8UC3, cv::Scalar(128));
    LabeledImage<BGR> a{Image<BGR>(m1), {1.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m2), {0.0f, 1.0f}};
    std::mt19937 rng(42);
    auto result = CutMix{}.p(1.0f)(a, b, rng);
    EXPECT_EQ(result.image.rows(), 80);
    EXPECT_EQ(result.image.cols(), 120);
    EXPECT_EQ(result.image.mat().type(), CV_8UC3);
}

TEST(MixAugTest, CutMixDeterministicSeed) {
    cv::Mat m1(50, 50, CV_8UC3, cv::Scalar(50));
    cv::Mat m2(50, 50, CV_8UC3, cv::Scalar(200));
    LabeledImage<BGR> a{Image<BGR>(m1), {1.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m2), {0.0f, 1.0f}};
    std::mt19937 rng1(7), rng2(7);
    auto r1 = CutMix{}.alpha(1.0f)(a, b, rng1);
    auto r2 = CutMix{}.alpha(1.0f)(a, b, rng2);
    EXPECT_EQ(cv::norm(r1.image.mat(), r2.image.mat(), cv::NORM_INF), 0.0);
    EXPECT_EQ(r1.label, r2.label);
}

TEST(MixAugTest, CutMixThrowsOnSizeMismatch) {
    cv::Mat m1(100, 100, CV_8UC3, cv::Scalar(0));
    cv::Mat m2(200, 200, CV_8UC3, cv::Scalar(0));
    LabeledImage<BGR> a{Image<BGR>(m1), {1.0f, 0.0f}};
    LabeledImage<BGR> b{Image<BGR>(m2), {0.0f, 1.0f}};
    std::mt19937 rng(0);
    EXPECT_THROW(CutMix{}.p(1.0f)(a, b, rng), ParameterError);
}
