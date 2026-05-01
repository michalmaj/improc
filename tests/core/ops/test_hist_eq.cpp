// tests/core/ops/test_hist_eq.cpp
#include <gtest/gtest.h>
#include "improc/core/ops/hist_eq.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

TEST(HistogramEqualizationTest, GrayDefaultPreservesSizeAndType) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(50)));
    auto result = HistogramEqualization{}(img);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(HistogramEqualizationTest, BGRDefaultPreservesSizeAndType) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(30, 40, 50)));
    auto result = HistogramEqualization{}(img);
    EXPECT_EQ(result.rows(), 32);
    EXPECT_EQ(result.cols(), 32);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(HistogramEqualizationTest, GrayShiftsMeanTowardMiddle) {
    // Dark-biased gradient [0..63] repeated — mean ~31; equalization spreads to full range → mean ~127
    cv::Mat mat(256, 1, CV_8UC1);
    for (int i = 0; i < 256; ++i) mat.at<uchar>(i, 0) = static_cast<uchar>(i % 64);
    Image<Gray> img(mat);
    cv::Scalar mean_before = cv::mean(mat);
    auto result = HistogramEqualization{}(img);
    cv::Scalar mean_after = cv::mean(result.mat());
    EXPECT_GT(mean_after[0], mean_before[0]);
}

TEST(HistogramEqualizationTest, BGRShiftsLuminanceMean) {
    // Dark-biased BGR: Y channel is concentrated in low values → equalization raises Y mean
    // Build a BGR image whose Y (luma) values span [0..63] — dark biased
    cv::Mat mat(256, 1, CV_8UC3);
    for (int i = 0; i < 256; ++i) {
        uchar v = static_cast<uchar>(i % 64);
        mat.at<cv::Vec3b>(i, 0) = cv::Vec3b(v, v, v);  // equal channels → Y ≈ v
    }
    Image<BGR> img(mat);

    cv::Mat ycrcb_before;
    cv::cvtColor(mat, ycrcb_before, cv::COLOR_BGR2YCrCb);
    cv::Scalar mean_before = cv::mean(ycrcb_before);  // [0] = Y channel mean

    auto result = HistogramEqualization{}(img);

    cv::Mat ycrcb_after;
    cv::cvtColor(result.mat(), ycrcb_after, cv::COLOR_BGR2YCrCb);
    cv::Scalar mean_after = cv::mean(ycrcb_after);
    EXPECT_GT(mean_after[0], mean_before[0]);
}

TEST(HistogramEqualizationTest, GrayPipelineSyntax) {
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(50)));
    auto result = img | HistogramEqualization{};
    EXPECT_EQ(result.mat().type(), CV_8UC1);
}

TEST(HistogramEqualizationTest, BGRPipelineSyntax) {
    Image<BGR> img(cv::Mat(32, 32, CV_8UC3, cv::Scalar(30, 40, 50)));
    auto result = img | HistogramEqualization{};
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(HistogramEqualizationTest, GrayUniformImagePreservesType) {
    // A uniform image is valid input; equalizeHist maps all pixels to 255
    Image<Gray> img(cv::Mat(32, 32, CV_8UC1, cv::Scalar(128)));
    auto result = HistogramEqualization{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 32);
}

TEST(HistogramEqualizationTest, BGRPreservesSize) {
    Image<BGR> img(cv::Mat(64, 48, CV_8UC3, cv::Scalar(60, 80, 100)));
    auto result = HistogramEqualization{}(img);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 48);
}

TEST(HistogramEqualizationTest, GrayFullRangeImagePreservesType) {
    // Gradient 0..255 — full range; equalizeHist on a perfect gradient
    cv::Mat mat(256, 1, CV_8UC1);
    for (int i = 0; i < 256; ++i) mat.at<uchar>(i, 0) = static_cast<uchar>(i);
    Image<Gray> img(mat);
    auto result = HistogramEqualization{}(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 256);
}
