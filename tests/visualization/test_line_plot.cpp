// tests/visualization/test_line_plot.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/visualization/line_plot.hpp"

using namespace improc::core;
using namespace improc::visualization;

TEST(LinePlotTest, EmptyVectorThrows) {
    EXPECT_THROW(LinePlot{}({}), improc::ParameterError);
}

TEST(LinePlotTest, SingleValueDoesNotThrow) {
    EXPECT_NO_THROW(LinePlot{}({1.0f}));
}

TEST(LinePlotTest, DefaultOutputType) {
    Image<BGR> result = LinePlot{}({0.0f, 0.5f, 1.0f});
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(LinePlotTest, DefaultOutputSize) {
    Image<BGR> result = LinePlot{}({0.0f, 0.5f, 1.0f});
    EXPECT_EQ(result.cols(), 640);
    EXPECT_EQ(result.rows(), 360);
}

TEST(LinePlotTest, CustomSize) {
    Image<BGR> result = LinePlot{}.width(320).height(240)({0.0f, 1.0f});
    EXPECT_EQ(result.cols(), 320);
    EXPECT_EQ(result.rows(), 240);
}

TEST(LinePlotTest, ZeroWidthThrows) {
    EXPECT_THROW(LinePlot{}.width(0), improc::ParameterError);
}

TEST(LinePlotTest, NegativeWidthThrows) {
    EXPECT_THROW(LinePlot{}.width(-1), improc::ParameterError);
}

TEST(LinePlotTest, ZeroHeightThrows) {
    EXPECT_THROW(LinePlot{}.height(0), improc::ParameterError);
}

TEST(LinePlotTest, NegativeHeightThrows) {
    EXPECT_THROW(LinePlot{}.height(-1), improc::ParameterError);
}
