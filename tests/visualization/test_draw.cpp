// tests/visualization/test_draw.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/visualization/draw.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::visualization;
using namespace improc::core;
using improc::ml::Detection;

namespace {

Image<BGR> make_image(int w = 64, int h = 64) {
    return Image<BGR>(cv::Mat(h, w, CV_8UC3, cv::Scalar(0, 0, 0)));
}

Detection make_detection(float x, float y, float w, float h,
                          std::string label = "cat", float conf = 0.9f) {
    Detection d;
    d.box        = {x, y, w, h};
    d.class_id   = 0;
    d.confidence = conf;
    d.label      = std::move(label);
    return d;
}

} // namespace

TEST(DrawBoundingBoxesTest, ZeroThicknessThrows) {
    EXPECT_THROW(DrawBoundingBoxes{{}}.thickness(0), improc::ParameterError);
}
TEST(DrawBoundingBoxesTest, NegativeThicknessThrows) {
    EXPECT_THROW(DrawBoundingBoxes{{}}.thickness(-1), improc::ParameterError);
}
TEST(DrawBoundingBoxesTest, ZeroFontScaleThrows) {
    EXPECT_THROW(DrawBoundingBoxes{{}}.font_scale(0.0), improc::ParameterError);
}

TEST(DrawBoundingBoxesTest, EmptyDetectionsReturnsClone) {
    auto img    = make_image();
    auto result = DrawBoundingBoxes{{}}.show_label(false)(img);
    EXPECT_EQ(result.rows(), img.rows());
    EXPECT_EQ(result.cols(), img.cols());
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(DrawBoundingBoxesTest, OutputSizeMatchesInput) {
    std::vector<Detection> dets = {make_detection(5, 5, 20, 20)};
    auto img    = make_image();
    auto result = DrawBoundingBoxes{dets}(img);
    EXPECT_EQ(result.rows(), img.rows());
    EXPECT_EQ(result.cols(), img.cols());
}

TEST(DrawBoundingBoxesTest, DrawingModifiesPixels) {
    // Black canvas — after drawing a green box some pixels must be non-black
    std::vector<Detection> dets = {make_detection(5, 5, 30, 30)};
    auto img    = make_image();
    auto result = DrawBoundingBoxes{dets}.show_label(false).show_confidence(false)(img);
    cv::Mat gray;
    cv::cvtColor(result.mat(), gray, cv::COLOR_BGR2GRAY);
    EXPECT_GT(cv::countNonZero(gray), 0);
}

TEST(DrawBoundingBoxesTest, SourceImageUnmodified) {
    // operator() draws onto a clone — original must stay black
    std::vector<Detection> dets = {make_detection(5, 5, 30, 30)};
    auto img = make_image();
    DrawBoundingBoxes{dets}.show_label(false).show_confidence(false)(img);
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    EXPECT_EQ(cv::countNonZero(gray), 0);
}

TEST(DrawBoundingBoxesTest, PipelineForm) {
    std::vector<Detection> dets = {make_detection(5, 5, 20, 20, "dog", 0.85f)};
    auto img    = make_image();
    auto result = img | DrawBoundingBoxes{dets}.thickness(1);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}
