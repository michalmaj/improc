// tests/core/ops/test_drawing.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// ── DrawText ──────────────────────────────────────────────────────────────────

TEST(DrawTextTest, PreservesSizeAndType) {
    Image<BGR> src(cv::Mat(200, 300, CV_8UC3, cv::Scalar(0, 0, 0)));
    Image<BGR> result = src | DrawText{"hello"};
    EXPECT_EQ(result.rows(), 200);
    EXPECT_EQ(result.cols(), 300);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(DrawTextTest, ChangesPixels) {
    cv::Mat blank(200, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> src(blank.clone());
    Image<BGR> result = src | DrawText{"hello"};
    cv::Mat diff;
    cv::absdiff(blank, result.mat(), diff);
    EXPECT_GT(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(DrawTextTest, DoesNotMutateSource) {
    cv::Mat blank(200, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> src(blank.clone());
    src | DrawText{"hello"};
    double diff = cv::norm(blank, src.mat(), cv::NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

TEST(DrawTextTest, FontScaleZeroThrows) {
    EXPECT_THROW((DrawText{"hi"}.font_scale(0.0)), improc::ParameterError);
}

TEST(DrawTextTest, FontScaleNegativeThrows) {
    EXPECT_THROW((DrawText{"hi"}.font_scale(-1.0)), improc::ParameterError);
}

TEST(DrawTextTest, ThicknessZeroThrows) {
    EXPECT_THROW((DrawText{"hi"}.thickness(0)), improc::ParameterError);
}

TEST(DrawTextTest, ThicknessNegativeThrows) {
    EXPECT_THROW((DrawText{"hi"}.thickness(-1)), improc::ParameterError);
}

TEST(DrawTextTest, PipelineSyntax) {
    Image<BGR> src(cv::Mat(200, 300, CV_8UC3, cv::Scalar(50, 50, 50)));
    auto result = src | DrawText{"test"}.position({10, 50}).color({255, 0, 0}).font_scale(0.5);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

// ── DrawLine ──────────────────────────────────────────────────────────────────

TEST(DrawLineTest, PreservesSizeAndType) {
    Image<BGR> src(cv::Mat(200, 300, CV_8UC3, cv::Scalar(0, 0, 0)));
    Image<BGR> result = src | DrawLine{{10, 10}, {200, 100}};
    EXPECT_EQ(result.rows(), 200);
    EXPECT_EQ(result.cols(), 300);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(DrawLineTest, ChangesPixels) {
    cv::Mat blank(200, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> src(blank.clone());
    Image<BGR> result = src | DrawLine{{10, 10}, {200, 100}};
    cv::Mat diff;
    cv::absdiff(blank, result.mat(), diff);
    EXPECT_GT(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(DrawLineTest, DoesNotMutateSource) {
    cv::Mat blank(200, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> src(blank.clone());
    src | DrawLine{{10, 10}, {200, 100}};
    double diff = cv::norm(blank, src.mat(), cv::NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

TEST(DrawLineTest, ThicknessZeroThrows) {
    EXPECT_THROW((DrawLine{{0,0},{10,10}}.thickness(0)), improc::ParameterError);
}

TEST(DrawLineTest, ThicknessNegativeThrows) {
    EXPECT_THROW((DrawLine{{0,0},{10,10}}.thickness(-1)), improc::ParameterError);
}

TEST(DrawLineTest, PipelineSyntax) {
    Image<BGR> src(cv::Mat(200, 300, CV_8UC3, cv::Scalar(50, 50, 50)));
    auto result = src | DrawLine{{0, 0}, {299, 199}}.color({0, 0, 255}).thickness(2);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

// ── DrawCircle ────────────────────────────────────────────────────────────────

TEST(DrawCircleTest, PreservesSizeAndType) {
    Image<BGR> src(cv::Mat(200, 300, CV_8UC3, cv::Scalar(0, 0, 0)));
    Image<BGR> result = src | DrawCircle{{150, 100}, 50};
    EXPECT_EQ(result.rows(), 200);
    EXPECT_EQ(result.cols(), 300);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(DrawCircleTest, RadiusZeroThrows) {
    EXPECT_THROW((DrawCircle{{100, 100}, 0}), improc::ParameterError);
}

TEST(DrawCircleTest, RadiusNegativeThrows) {
    EXPECT_THROW((DrawCircle{{100, 100}, -5}), improc::ParameterError);
}

TEST(DrawCircleTest, ThicknessZeroThrows) {
    EXPECT_THROW((DrawCircle{{100, 100}, 30}.thickness(0)), improc::ParameterError);
}

TEST(DrawCircleTest, ThicknessMinusTwoThrows) {
    EXPECT_THROW((DrawCircle{{100, 100}, 30}.thickness(-2)), improc::ParameterError);
}

TEST(DrawCircleTest, ThicknessMinusOneWorks) {
    Image<BGR> src(cv::Mat(200, 300, CV_8UC3, cv::Scalar(0, 0, 0)));
    EXPECT_NO_THROW((src | DrawCircle{{150, 100}, 50}.thickness(-1)));
}

TEST(DrawCircleTest, DoesNotMutateSource) {
    cv::Mat blank(200, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> src(blank.clone());
    src | DrawCircle{{150, 100}, 50};
    double diff = cv::norm(blank, src.mat(), cv::NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

TEST(DrawCircleTest, PipelineSyntax) {
    Image<BGR> src(cv::Mat(200, 300, CV_8UC3, cv::Scalar(50, 50, 50)));
    auto result = src | DrawCircle{{150, 100}, 40}.color({255, 255, 0}).thickness(2);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

// ── DrawRectangle ─────────────────────────────────────────────────────────────

TEST(DrawRectangleTest, PreservesSizeAndType) {
    Image<BGR> src(cv::Mat(200, 300, CV_8UC3, cv::Scalar(0, 0, 0)));
    Image<BGR> result = src | DrawRectangle{cv::Rect{50, 50, 100, 80}};
    EXPECT_EQ(result.rows(), 200);
    EXPECT_EQ(result.cols(), 300);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(DrawRectangleTest, ChangesPixels) {
    cv::Mat blank(200, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> src(blank.clone());
    Image<BGR> result = src | DrawRectangle{cv::Rect{50, 50, 100, 80}};
    cv::Mat diff;
    cv::absdiff(blank, result.mat(), diff);
    EXPECT_GT(cv::countNonZero(diff.reshape(1)), 0);
}

TEST(DrawRectangleTest, DoesNotMutateSource) {
    cv::Mat blank(200, 300, CV_8UC3, cv::Scalar(0, 0, 0));
    Image<BGR> src(blank.clone());
    src | DrawRectangle{cv::Rect{50, 50, 100, 80}};
    double diff = cv::norm(blank, src.mat(), cv::NORM_INF);
    EXPECT_EQ(diff, 0.0);
}

TEST(DrawRectangleTest, ThicknessZeroThrows) {
    EXPECT_THROW((DrawRectangle{cv::Rect{10,10,50,50}}.thickness(0)), improc::ParameterError);
}

TEST(DrawRectangleTest, ThicknessMinusTwoThrows) {
    EXPECT_THROW((DrawRectangle{cv::Rect{10,10,50,50}}.thickness(-2)), improc::ParameterError);
}

TEST(DrawRectangleTest, ThicknessMinusOneWorks) {
    Image<BGR> src(cv::Mat(200, 300, CV_8UC3, cv::Scalar(0, 0, 0)));
    EXPECT_NO_THROW((src | DrawRectangle{cv::Rect{50, 50, 100, 80}}.thickness(-1)));
}

TEST(DrawRectangleTest, PipelineSyntax) {
    Image<BGR> src(cv::Mat(200, 300, CV_8UC3, cv::Scalar(50, 50, 50)));
    auto result = src | DrawRectangle{cv::Rect{10, 10, 100, 80}}.color({0, 255, 0}).thickness(2);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}
