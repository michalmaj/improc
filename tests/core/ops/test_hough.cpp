// tests/core/ops/test_hough.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/ops/hough.hpp"

using namespace improc::core;

namespace {
Image<Gray> make_line_image() {
    cv::Mat mat(100, 100, CV_8UC1, cv::Scalar(0));
    cv::line(mat, {0, 50}, {99, 50}, cv::Scalar(255), 1);
    return Image<Gray>(mat);
}

Image<Gray> make_circle_image() {
    cv::Mat mat(200, 200, CV_8UC1, cv::Scalar(0));
    cv::circle(mat, {100, 100}, 40, cv::Scalar(255), 2);
    return Image<Gray>(mat);
}

Image<Gray> make_blank_image() {
    return Image<Gray>(cv::Mat(100, 100, CV_8UC1, cv::Scalar(0)));
}
} // namespace

// --- HoughLinesP ---

TEST(HoughLinesPTest, DefaultConstruction) {
    EXPECT_NO_THROW(HoughLinesP{});
}

TEST(HoughLinesPTest, FluentSetterChain) {
    HoughLinesP op;
    HoughLinesP& ref = op.rho(1.0).threshold(50);
    EXPECT_EQ(&ref, &op);
}

TEST(HoughLinesPTest, DetectsHorizontalLine) {
    auto lines = HoughLinesP{}.threshold(30).min_line_length(50.0)(make_line_image());
    EXPECT_GE(lines.size(), 1u);
}

TEST(HoughLinesPTest, BlankImageReturnsEmpty) {
    auto lines = HoughLinesP{}(make_blank_image());
    EXPECT_TRUE(lines.empty());
}

// --- HoughCircles ---

TEST(HoughCirclesTest, DefaultConstruction) {
    EXPECT_NO_THROW(HoughCircles{});
}

TEST(HoughCirclesTest, FluentSetterChain) {
    HoughCircles op;
    HoughCircles& ref = op.param2(20.0).min_radius(10);
    EXPECT_EQ(&ref, &op);
}

TEST(HoughCirclesTest, DetectsCircleWithMatchingRadius) {
    auto circles = HoughCircles{}.param2(20.0).min_radius(30).max_radius(60)(make_circle_image());
    ASSERT_GE(circles.size(), 1u);
    EXPECT_NEAR(circles[0][2], 40.0f, 2.0f);
}

TEST(HoughCirclesTest, BlankImageReturnsEmpty) {
    auto circles = HoughCircles{}(make_blank_image());
    EXPECT_TRUE(circles.empty());
}
