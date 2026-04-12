// tests/core/ops/test_normalize.cpp
#include <gtest/gtest.h>
#include "improc/core/ops/normalize.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// --- Normalize (auto min-max) ---

TEST(NormalizeTest, ScalesToZeroOne) {
    cv::Mat mat(1, 2, CV_32FC1);
    mat.at<float>(0, 0) = 0.0f;
    mat.at<float>(0, 1) = 255.0f;
    Image<Float32> img(mat);
    Image<Float32> result = img | Normalize{};
    EXPECT_NEAR(result.mat().at<float>(0, 0), 0.0f, 1e-5f);
    EXPECT_NEAR(result.mat().at<float>(0, 1), 1.0f, 1e-5f);
}

TEST(NormalizeTest, UniformImageReturnsZeros) {
    cv::Mat mat(2, 2, CV_32FC1, cv::Scalar(42.0f));
    Image<Float32> img(mat);
    Image<Float32> result = img | Normalize{};
    EXPECT_NEAR(result.mat().at<float>(0, 0), 0.0f, 1e-5f);
}

TEST(NormalizeTest, PreservesSize) {
    cv::Mat mat(10, 20, CV_32FC1);
    mat.setTo(1.0f);
    Image<Float32> img(mat);
    Image<Float32> result = img | Normalize{};
    EXPECT_EQ(result.rows(), 10);
    EXPECT_EQ(result.cols(), 20);
}

// --- NormalizeTo ---

TEST(NormalizeToTest, ScalesToExplicitRange) {
    cv::Mat mat(1, 2, CV_32FC1);
    mat.at<float>(0, 0) = 0.0f;
    mat.at<float>(0, 1) = 100.0f;
    Image<Float32> img(mat);
    Image<Float32> result = img | NormalizeTo{-1.0f, 1.0f};
    EXPECT_NEAR(result.mat().at<float>(0, 0), -1.0f, 1e-5f);
    EXPECT_NEAR(result.mat().at<float>(0, 1),  1.0f, 1e-5f);
}

TEST(NormalizeToTest, ThrowsOnInvalidRange) {
    EXPECT_THROW(NormalizeTo(1.0f, 0.0f), std::invalid_argument);
    EXPECT_THROW(NormalizeTo(1.0f, 1.0f), std::invalid_argument);
}

TEST(NormalizeToTest, UniformImageReturnsZeros) {
    // spec does not define this case; behavior: returns zero-filled image (same as Normalize)
    cv::Mat mat(2, 2, CV_32FC1, cv::Scalar(5.0f));
    Image<Float32> img(mat);
    Image<Float32> result = img | NormalizeTo{0.0f, 1.0f};
    EXPECT_NEAR(result.mat().at<float>(0, 0), 0.0f, 1e-5f);
}

// --- Standardize ---

TEST(StandardizeTest, AppliesZScore) {
    cv::Mat mat(1, 2, CV_32FC1);
    mat.at<float>(0, 0) = 2.0f;
    mat.at<float>(0, 1) = 4.0f;
    Image<Float32> img(mat);
    Image<Float32> result = img | Standardize{3.0f, 1.0f};  // mean=3, std=1
    EXPECT_NEAR(result.mat().at<float>(0, 0), -1.0f, 1e-5f);  // (2-3)/1
    EXPECT_NEAR(result.mat().at<float>(0, 1),  1.0f, 1e-5f);  // (4-3)/1
}

TEST(StandardizeTest, ThrowsOnZeroStdDev) {
    EXPECT_THROW(Standardize(0.5f, 0.0f),  std::invalid_argument);
    EXPECT_THROW(Standardize(0.5f, -1.0f), std::invalid_argument);
}

// --- Float32C3 overloads (global normalization across all channels) ---

TEST(NormalizeFloat32C3Test, ScalesToZeroOne) {
    cv::Mat mat(1, 2, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
    mat.at<cv::Vec3f>(0, 1) = {255.0f, 255.0f, 255.0f};
    Image<Float32C3> img(mat);
    Image<Float32C3> result = img | Normalize{};
    EXPECT_NEAR(result.mat().at<cv::Vec3f>(0, 0)[0], 0.0f, 1e-5f);
    EXPECT_NEAR(result.mat().at<cv::Vec3f>(0, 1)[0], 1.0f, 1e-5f);
    EXPECT_EQ(result.mat().type(), CV_32FC3);
}

TEST(NormalizeFloat32C3Test, UniformImageReturnsZeros) {
    cv::Mat mat(2, 2, CV_32FC3, cv::Scalar(42.0f, 42.0f, 42.0f));
    Image<Float32C3> img(mat);
    Image<Float32C3> result = img | Normalize{};
    EXPECT_NEAR(result.mat().at<cv::Vec3f>(0, 0)[0], 0.0f, 1e-5f);
}

TEST(NormalizeToFloat32C3Test, ScalesToExplicitRange) {
    cv::Mat mat(1, 2, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));
    mat.at<cv::Vec3f>(0, 1) = {100.0f, 100.0f, 100.0f};
    Image<Float32C3> img(mat);
    Image<Float32C3> result = img | NormalizeTo{-1.0f, 1.0f};
    EXPECT_NEAR(result.mat().at<cv::Vec3f>(0, 0)[0], -1.0f, 1e-5f);
    EXPECT_NEAR(result.mat().at<cv::Vec3f>(0, 1)[0],  1.0f, 1e-5f);
}

TEST(StandardizeFloat32C3Test, AppliesZScore) {
    cv::Mat mat(1, 2, CV_32FC3);
    mat.at<cv::Vec3f>(0, 0) = {2.0f, 2.0f, 2.0f};
    mat.at<cv::Vec3f>(0, 1) = {4.0f, 4.0f, 4.0f};
    Image<Float32C3> img(mat);
    Image<Float32C3> result = img | Standardize{3.0f, 1.0f};
    EXPECT_NEAR(result.mat().at<cv::Vec3f>(0, 0)[0], -1.0f, 1e-5f);
    EXPECT_NEAR(result.mat().at<cv::Vec3f>(0, 1)[0],  1.0f, 1e-5f);
    EXPECT_EQ(result.mat().type(), CV_32FC3);
}
