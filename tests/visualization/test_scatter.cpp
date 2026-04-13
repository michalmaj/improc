// tests/visualization/test_scatter.cpp
#include <gtest/gtest.h>
#include "improc/visualization/scatter.hpp"

using namespace improc::core;
using namespace improc::visualization;

TEST(ScatterTest, EmptyXsThrows) {
    EXPECT_THROW(Scatter{}({}, {1.0f}), std::invalid_argument);
}

TEST(ScatterTest, EmptyYsThrows) {
    EXPECT_THROW(Scatter{}({1.0f}, {}), std::invalid_argument);
}

TEST(ScatterTest, MismatchedSizesThrow) {
    EXPECT_THROW(Scatter{}({1.0f, 2.0f}, {1.0f}), std::invalid_argument);
}

TEST(ScatterTest, SinglePointDoesNotThrow) {
    EXPECT_NO_THROW(Scatter{}({0.5f}, {0.5f}));
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
    EXPECT_THROW(Scatter{}.point_radius(0), std::invalid_argument);
}

TEST(ScatterTest, NegativeWidthThrows) {
    EXPECT_THROW(Scatter{}.width(-1), std::invalid_argument);
}

TEST(ScatterTest, NegativeHeightThrows) {
    EXPECT_THROW(Scatter{}.height(-1), std::invalid_argument);
}
