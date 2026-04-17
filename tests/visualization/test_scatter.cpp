// tests/visualization/test_scatter.cpp
#include <gtest/gtest.h>
#include "improc/exceptions.hpp"
#include "improc/visualization/scatter.hpp"

using namespace improc::core;
using namespace improc::visualization;

TEST(ScatterTest, EmptyXsThrows) {
    EXPECT_THROW(Scatter{}({}, {1.0f}), improc::ParameterError);
}

TEST(ScatterTest, EmptyYsThrows) {
    EXPECT_THROW(Scatter{}({1.0f}, {}), improc::ParameterError);
}

TEST(ScatterTest, MismatchedSizesThrow) {
    EXPECT_THROW(Scatter{}({1.0f, 2.0f}, {1.0f}), improc::ParameterError);
}

TEST(ScatterTest, SinglePointDoesNotThrow) {
    Image<BGR> result = Scatter{}({0.5f}, {0.5f});
    EXPECT_EQ(result.mat().type(), CV_8UC3);
    EXPECT_EQ(result.cols(), 512);
    EXPECT_EQ(result.rows(), 512);
}

TEST(ScatterTest, DefaultOutputType) {
    Image<BGR> result = Scatter{}({0.0f, 1.0f}, {0.0f, 1.0f});
    EXPECT_EQ(result.mat().type(), CV_8UC3);
}

TEST(ScatterTest, DefaultOutputSize) {
    Image<BGR> result = Scatter{}({0.0f, 1.0f}, {0.0f, 1.0f});
    EXPECT_EQ(result.cols(), 512);
    EXPECT_EQ(result.rows(), 512);
}

TEST(ScatterTest, CustomSize) {
    Image<BGR> result = Scatter{}.width(300).height(300)({0.0f, 1.0f}, {0.0f, 1.0f});
    EXPECT_EQ(result.cols(), 300);
    EXPECT_EQ(result.rows(), 300);
}

TEST(ScatterTest, ZeroPointRadiusThrows) {
    EXPECT_THROW(Scatter{}.point_radius(0), improc::ParameterError);
}

TEST(ScatterTest, NegativeWidthThrows) {
    EXPECT_THROW(Scatter{}.width(-1), improc::ParameterError);
}

TEST(ScatterTest, NegativeHeightThrows) {
    EXPECT_THROW(Scatter{}.height(-1), improc::ParameterError);
}

TEST(ScatterTest, ZeroWidthThrows) {
    EXPECT_THROW(Scatter{}.width(0), improc::ParameterError);
}

TEST(ScatterTest, ZeroHeightThrows) {
    EXPECT_THROW(Scatter{}.height(0), improc::ParameterError);
}

TEST(ScatterTest, NegativePointRadiusThrows) {
    EXPECT_THROW(Scatter{}.point_radius(-1), improc::ParameterError);
}
