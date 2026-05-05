// tests/core/ops/test_contours.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// helper: binary image with a white rectangle on black background
static Image<Gray> make_rect_binary() {
    cv::Mat m(100, 100, CV_8UC1, cv::Scalar(0));
    cv::rectangle(m, {20, 20}, {79, 79}, cv::Scalar(255), -1);
    return Image<Gray>(m);
}

// ── ContourSet ────────────────────────────────────────────────────────────────

TEST(ContourSetTest, DefaultIsEmpty) {
    ContourSet cs;
    EXPECT_EQ(cs.size(), 0u);
    EXPECT_TRUE(cs.contours.empty());
    EXPECT_TRUE(cs.hierarchy.empty());
}

TEST(ContourSetTest, SizeMatchesContours) {
    Image<Gray> src = make_rect_binary();
    ContourSet cs = src | FindContours{};
    EXPECT_EQ(cs.size(), cs.contours.size());
}

TEST(ContourSetTest, AreaIsPositiveForRectangle) {
    Image<Gray> src = make_rect_binary();
    ContourSet cs = src | FindContours{};
    ASSERT_GT(cs.size(), 0u);
    EXPECT_GT(cs.area(0), 0.0);
}

TEST(ContourSetTest, PerimeterIsPositiveForRectangle) {
    Image<Gray> src = make_rect_binary();
    ContourSet cs = src | FindContours{};
    ASSERT_GT(cs.size(), 0u);
    EXPECT_GT(cs.perimeter(0), 0.0);
}

TEST(ContourSetTest, BoundingRectIsNonEmpty) {
    Image<Gray> src = make_rect_binary();
    ContourSet cs = src | FindContours{};
    ASSERT_GT(cs.size(), 0u);
    cv::Rect br = cs.bounding_rect(0);
    EXPECT_GT(br.width, 0);
    EXPECT_GT(br.height, 0);
}

// ── FindContours ──────────────────────────────────────────────────────────────

TEST(FindContoursTest, FindsContoursInBinaryImage) {
    Image<Gray> src = make_rect_binary();
    ContourSet cs = src | FindContours{};
    EXPECT_GT(cs.size(), 0u);
}

TEST(FindContoursTest, EmptyImageHasNoContours) {
    Image<Gray> src(cv::Mat(50, 50, CV_8UC1, cv::Scalar(0)));
    ContourSet cs = src | FindContours{};
    EXPECT_EQ(cs.size(), 0u);
}

TEST(FindContoursTest, PipelineSyntax) {
    Image<Gray> src = make_rect_binary();
    auto cs = src | FindContours{};
    EXPECT_GT(cs.size(), 0u);
}

TEST(FindContoursTest, ModeExternalVsList) {
    // List mode finds at least as many contours as External
    Image<Gray> src = make_rect_binary();
    ContourSet ext  = src | FindContours{}.mode(FindContours::Mode::External);
    ContourSet list = src | FindContours{}.mode(FindContours::Mode::List);
    EXPECT_GE(list.size(), ext.size());
}

TEST(FindContoursTest, MethodSimpleVsNone) {
    // Simple compresses horizontal/vertical/diagonal segments; None keeps all points.
    // None should produce >= as many contour points as Simple.
    Image<Gray> src = make_rect_binary();
    ContourSet simple = src | FindContours{}.method(FindContours::Method::Simple);
    ContourSet none   = src | FindContours{}.method(FindContours::Method::None);
    ASSERT_GT(simple.size(), 0u);
    ASSERT_GT(none.size(), 0u);
    EXPECT_GE(none.contours[0].size(), simple.contours[0].size());
}

// ── DrawContours ──────────────────────────────────────────────────────────────

TEST(DrawContoursTest, PreservesSizeAndType) {
    Image<Gray> gray = make_rect_binary();
    ContourSet cs = gray | FindContours{};
    Image<BGR> bgr(cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0)));
    Image<BGR> result = bgr | DrawContours{cs};
    EXPECT_EQ(result.rows(), 100);
    EXPECT_EQ(result.cols(), 100);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(DrawContoursTest, ChangesPixels) {
    Image<Gray> gray = make_rect_binary();
    ContourSet cs = gray | FindContours{};
    cv::Mat blank(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> bgr(blank.clone());
    Image<BGR> result = bgr | DrawContours{cs};
    cv::Mat diff;
    cv::absdiff(blank, result.mat(), diff);
    EXPECT_GT(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(DrawContoursTest, DoesNotMutateSource) {
    Image<Gray> gray = make_rect_binary();
    ContourSet cs = gray | FindContours{};
    cv::Mat blank(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> bgr(blank.clone());
    bgr | DrawContours{cs};
    double diff = cv::norm(blank, bgr.mat(), cv::NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

TEST(DrawContoursTest, ThicknessZeroThrows) {
    ContourSet cs;
    EXPECT_THROW((DrawContours{cs}.thickness(0)), improc::ParameterError);
}

TEST(DrawContoursTest, ThicknessMinusTwoThrows) {
    ContourSet cs;
    EXPECT_THROW((DrawContours{cs}.thickness(-2)), improc::ParameterError);
}

TEST(DrawContoursTest, ThicknessMinusOneWorks) {
    Image<Gray> gray = make_rect_binary();
    ContourSet cs = gray | FindContours{};
    Image<BGR> bgr(cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0)));
    EXPECT_NO_THROW((bgr | DrawContours{cs}.thickness(-1)));
}

TEST(DrawContoursTest, PipelineSyntax) {
    Image<Gray> gray = make_rect_binary();
    ContourSet cs = gray | FindContours{};
    Image<BGR> bgr(cv::Mat(100, 100, CV_8UC3, cv::Scalar(50, 50, 50)));
    auto result = bgr | DrawContours{cs}.color({0, 0, 255}).index(-1);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(DrawContoursTest, DrawSingleContourByIndex) {
    Image<Gray> gray = make_rect_binary();
    ContourSet cs = gray | FindContours{};
    ASSERT_GT(cs.size(), 0u);
    Image<BGR> bgr(cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0)));
    EXPECT_NO_THROW((bgr | DrawContours{cs}.index(0)));
}
