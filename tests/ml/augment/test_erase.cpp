// tests/ml/augment/test_erase.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include <opencv2/core.hpp>
#include "improc/ml/augment/erase.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/ml/augment/compose.hpp"

using namespace improc::core;
using namespace improc::ml;

// ---- RandomErasing ----

TEST(EraseAugTest, RandomErasingPreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 150, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomErasing{}.p(1.0f)(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(EraseAugTest, RandomErasingGrayPreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC1, cv::Scalar(128));
    Image<Gray> img(mat);
    std::mt19937 rng(42);
    Image<Gray> result = RandomErasing{}.p(1.0f)(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(EraseAugTest, RandomErasingP0Identity) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomErasing{}.p(0.0f)(img, rng);
    cv::Mat diff;
    cv::absdiff(img.mat(), result.mat(), diff);
    EXPECT_EQ(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(EraseAugTest, RandomErasingP1ModifiesImage) {
    cv::Mat mat(128, 128, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = RandomErasing{}.p(1.0f)(img, rng);
    cv::Mat diff;
    cv::absdiff(img.mat(), result.mat(), diff);
    EXPECT_GT(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(EraseAugTest, RandomErasingInvalidPThrows) {
    EXPECT_THROW(RandomErasing{}.p(-0.1f), improc::ParameterError);
    EXPECT_THROW(RandomErasing{}.p(1.1f),  improc::ParameterError);
}

TEST(EraseAugTest, RandomErasingInvalidScaleThrows) {
    EXPECT_THROW(RandomErasing{}.scale(0.0f, 0.5f),  improc::ParameterError); // min <= 0
    EXPECT_THROW(RandomErasing{}.scale(0.1f, 1.1f),  improc::ParameterError); // max > 1
    EXPECT_THROW(RandomErasing{}.scale(0.5f, 0.1f),  improc::ParameterError); // min > max
}

TEST(EraseAugTest, RandomErasingInvalidRatioThrows) {
    EXPECT_THROW(RandomErasing{}.ratio(0.0f, 1.0f),  improc::ParameterError); // min <= 0
    EXPECT_THROW(RandomErasing{}.ratio(2.0f, 1.0f),  improc::ParameterError); // min > max
}

TEST(EraseAugTest, RandomErasingInvalidValueThrows) {
    EXPECT_THROW(RandomErasing{}.value(-1),  improc::ParameterError);
    EXPECT_THROW(RandomErasing{}.value(256), improc::ParameterError);
}

TEST(EraseAugTest, RandomErasingBindRngPipelineOp) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | RandomErasing{}.p(0.0f).bind(rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

// ---- GridDropout ----

TEST(EraseAugTest, GridDropoutPreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = GridDropout{}(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(EraseAugTest, GridDropoutGrayPreservesSizeAndType) {
    cv::Mat mat(64, 64, CV_8UC1, cv::Scalar(200));
    Image<Gray> img(mat);
    std::mt19937 rng(42);
    Image<Gray> result = GridDropout{}(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(EraseAugTest, GridDropoutHighRatioModifiesImage) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = GridDropout{}.ratio(0.99f).unit_size(8)(img, rng);
    cv::Mat diff;
    cv::absdiff(img.mat(), result.mat(), diff);
    EXPECT_GT(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(EraseAugTest, GridDropoutInvalidRatioThrows) {
    EXPECT_THROW(GridDropout{}.ratio(0.0f),  improc::ParameterError); // <= 0
    EXPECT_THROW(GridDropout{}.ratio(1.0f),  improc::ParameterError); // >= 1
}

TEST(EraseAugTest, GridDropoutInvalidUnitSizeThrows) {
    EXPECT_THROW(GridDropout{}.unit_size(0),  improc::ParameterError);
    EXPECT_THROW(GridDropout{}.unit_size(-1), improc::ParameterError);
}

TEST(EraseAugTest, GridDropoutInvalidValueThrows) {
    EXPECT_THROW(GridDropout{}.value(-1),  improc::ParameterError);
    EXPECT_THROW(GridDropout{}.value(256), improc::ParameterError);
}

TEST(EraseAugTest, GridDropoutBindRngPipelineOp) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    Image<BGR> result = img | GridDropout{}.ratio(0.5f).bind(rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

// ---- Compose integration ----

TEST(EraseAugTest, ComposeEraseOps) {
    cv::Mat mat(64, 64, CV_8UC3, cv::Scalar(200, 200, 200));
    Image<BGR> img(mat);
    std::mt19937 rng(42);
    auto pipeline = Compose<BGR>{}
        .add(RandomErasing{}.p(0.5f))
        .add(GridDropout{}.ratio(0.3f).unit_size(16));
    Image<BGR> result = pipeline(img, rng);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}
