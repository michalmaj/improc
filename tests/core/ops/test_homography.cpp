// tests/core/ops/test_homography.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/ops/homography.hpp"
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"
#include "improc/error.hpp"

using namespace improc::core;

// ── find_homography ───────────────────────────────────────────────────────────

TEST(FindHomographyTest, EmptyPointsReturnsInsufficientPoints) {
    std::vector<cv::Point2f> empty{};
    auto result = find_homography(empty, empty);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, improc::Error::Code::InsufficientPoints);
}

TEST(FindHomographyTest, FewerThanFourPointsReturnsError) {
    std::vector<cv::Point2f> src = {{0,0},{100,0},{100,100}};
    std::vector<cv::Point2f> dst = {{10,10},{110,10},{110,110}};
    auto result = find_homography(src, dst);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, improc::Error::Code::InsufficientPoints);
}

TEST(FindHomographyTest, MismatchedSizesReturnsError) {
    std::vector<cv::Point2f> src = {{0,0},{100,0},{100,100},{0,100}};
    std::vector<cv::Point2f> dst = {{10,10},{110,10},{110,110}};
    auto result = find_homography(src, dst);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, improc::Error::Code::InsufficientPoints);
}

TEST(FindHomographyTest, FourPointsReturns3x3Matrix) {
    std::vector<cv::Point2f> src = {{0,0},{100,0},{100,100},{0,100}};
    std::vector<cv::Point2f> dst = {{10,10},{110,10},{110,110},{10,110}};
    auto result = find_homography(src, dst);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows, 3);
    EXPECT_EQ(result->cols, 3);
    EXPECT_EQ(result->type(), CV_64F);
}

TEST(FindHomographyTest, RansacHandlesOutliers) {
    std::vector<cv::Point2f> src = {
        {0,0},{200,0},{200,200},{0,200},
        {50,50},{150,50},{150,150},{50,150}
    };
    std::vector<cv::Point2f> dst = {
        {0,0},{200,0},{200,200},{0,200},
        {999,999},{888,888},{777,777},{666,666}
    };
    auto result = find_homography(src, dst);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->rows, 3);
}

TEST(FindHomographyTest, CustomThresholdIsAccepted) {
    std::vector<cv::Point2f> src = {{0,0},{100,0},{100,100},{0,100}};
    std::vector<cv::Point2f> dst = {{0,0},{100,0},{100,100},{0,100}};
    auto result = find_homography(src, dst, 5.0);
    ASSERT_TRUE(result.has_value());
}

// ── WarpPerspective ───────────────────────────────────────────────────────────

TEST(WarpPerspectiveTest, NoHomographySetThrows) {
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100)));
    EXPECT_THROW(WarpPerspective{}(img), improc::ParameterError);
}

TEST(WarpPerspectiveTest, IdentityPreservesDimensions) {
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100)));
    auto result = WarpPerspective{}.homography(H)(img);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

TEST(WarpPerspectiveTest, CustomSizeApplied) {
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100)));
    auto result = WarpPerspective{}.homography(H).width(128).height(96)(img);
    EXPECT_EQ(result.rows(), 96);
    EXPECT_EQ(result.cols(), 128);
}

TEST(WarpPerspectiveTest, GrayFormatPreservesType) {
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    Image<Gray> img(cv::Mat(64, 64, CV_8UC1, cv::Scalar(128)));
    auto result = WarpPerspective{}.homography(H)(img);
    EXPECT_EQ(result.mat().type(), CV_8UC1);
    EXPECT_EQ(result.rows(), 64);
}

TEST(WarpPerspectiveTest, Float32FormatPreservesType) {
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    Image<Float32> img(cv::Mat(64, 64, CV_32FC1, cv::Scalar(0.5f)));
    auto result = WarpPerspective{}.homography(H)(img);
    EXPECT_EQ(result.mat().type(), CV_32FC1);
}

TEST(WarpPerspectiveTest, PipelineFormComposes) {
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(100, 100, 100)));
    auto result = img | WarpPerspective{}.homography(H);
    EXPECT_EQ(result.rows(), 64);
    EXPECT_EQ(result.cols(), 64);
}

TEST(WarpPerspectiveTest, TranslationShiftsContent) {
    cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
    H.at<double>(0, 2) = 10.0;
    Image<BGR> img(cv::Mat(64, 64, CV_8UC3, cv::Scalar(200, 100, 50)));
    auto result = WarpPerspective{}.homography(H)(img);
    cv::Vec3b topleft = result.mat().at<cv::Vec3b>(0, 0);
    EXPECT_EQ(topleft[0], 0);
    EXPECT_EQ(topleft[1], 0);
    EXPECT_EQ(topleft[2], 0);
}
