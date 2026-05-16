// tests/visualization/test_draw_tracks.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/visualization/draw_tracks.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::visualization;
using namespace improc::core;
using improc::ml::Track;
using improc::ml::BBox;

namespace {

Image<BGR> make_image(int w = 64, int h = 64) {
    return Image<BGR>(cv::Mat(h, w, CV_8UC3, cv::Scalar(0, 0, 0)));
}

Track make_track(int id, float x, float y, float w, float h) {
    Track t;
    t.id   = id;
    t.bbox = BBox{cv::Rect2f(x, y, w, h), 0, ""};
    return t;
}

} // namespace

TEST(DrawTracksTest, ZeroThicknessThrows) {
    EXPECT_THROW(DrawTracks{{}}.thickness(0), improc::ParameterError);
}
TEST(DrawTracksTest, NegativeThicknessThrows) {
    EXPECT_THROW(DrawTracks{{}}.thickness(-1), improc::ParameterError);
}
TEST(DrawTracksTest, ZeroFontScaleThrows) {
    EXPECT_THROW(DrawTracks{{}}.font_scale(0.0), improc::ParameterError);
}

TEST(DrawTracksTest, EmptyTracksReturnsClone) {
    auto img    = make_image();
    auto result = DrawTracks{{}}.show_id(false)(img);
    EXPECT_EQ(result.rows(), img.rows());
    EXPECT_EQ(result.cols(), img.cols());
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(DrawTracksTest, OutputSizeMatchesInput) {
    auto img    = make_image();
    auto result = DrawTracks{{make_track(0, 5, 5, 20, 20)}}(img);
    EXPECT_EQ(result.rows(), img.rows());
    EXPECT_EQ(result.cols(), img.cols());
}

TEST(DrawTracksTest, DrawingModifiesPixels) {
    auto img    = make_image();
    auto result = DrawTracks{{make_track(0, 5, 5, 30, 30)}}.show_id(false)(img);
    cv::Mat gray;
    cv::cvtColor(result.mat(), gray, cv::COLOR_BGR2GRAY);
    EXPECT_GT(cv::countNonZero(gray), 0);
}

TEST(DrawTracksTest, SourceImageUnmodified) {
    auto img = make_image();
    DrawTracks{{make_track(0, 5, 5, 30, 30)}}.show_id(false)(img);
    cv::Mat gray;
    cv::cvtColor(img.mat(), gray, cv::COLOR_BGR2GRAY);
    EXPECT_EQ(cv::countNonZero(gray), 0);
}

TEST(DrawTracksTest, PipelineForm) {
    auto img    = make_image();
    auto result = img | DrawTracks{{make_track(0, 5, 5, 20, 20)}}.thickness(1);
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}
