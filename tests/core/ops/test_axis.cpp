// tests/core/ops/test_axis.cpp
#include <gtest/gtest.h>
#include <string>
#include <opencv2/core.hpp>
#include "improc/core/ops/axis.hpp"
#include "improc/core/ops/flip.hpp"
#include "improc/core/pipeline.hpp"

using namespace improc::core;

// ── Enum usability ────────────────────────────────────────────────────────────

static std::string axis_name(Axis a) {
    switch (a) {
        case Axis::Horizontal: return "Horizontal";
        case Axis::Vertical:   return "Vertical";
        case Axis::Both:       return "Both";
    }
    return "unknown";
}

TEST(AxisTest, AllValuesAreNameable) {
    EXPECT_EQ(axis_name(Axis::Horizontal), "Horizontal");
    EXPECT_EQ(axis_name(Axis::Vertical),   "Vertical");
    EXPECT_EQ(axis_name(Axis::Both),       "Both");
}

TEST(AxisTest, ThreeDistinctValues) {
    EXPECT_NE(Axis::Horizontal, Axis::Vertical);
    EXPECT_NE(Axis::Horizontal, Axis::Both);
    EXPECT_NE(Axis::Vertical,   Axis::Both);
}

// ── Flip composition ──────────────────────────────────────────────────────────

TEST(AxisTest, HorizontalFlipIsNotSameAsVertical) {
    cv::Mat m(2, 2, CV_8UC3, cv::Scalar(0));
    m.at<cv::Vec3b>(0, 0) = {255, 0, 0};
    Image<BGR> img(m);

    Image<BGR> h = img | Flip{Axis::Horizontal};
    Image<BGR> v = img | Flip{Axis::Vertical};

    // Horizontal: pixel moves to top-right
    EXPECT_EQ(h.mat().at<cv::Vec3b>(0, 1), (cv::Vec3b{255, 0, 0}));
    // Vertical: pixel moves to bottom-left
    EXPECT_EQ(v.mat().at<cv::Vec3b>(1, 0), (cv::Vec3b{255, 0, 0}));
    // Results differ
    EXPECT_NE(cv::norm(h.mat(), v.mat(), cv::NORM_INF), 0.0);
}

TEST(AxisTest, BothAxesIsCompositionOfHorizontalAndVertical) {
    cv::Mat m(2, 2, CV_8UC3, cv::Scalar(0));
    m.at<cv::Vec3b>(0, 0) = {0, 128, 255};
    Image<BGR> img(m);

    Image<BGR> both      = img | Flip{Axis::Both};
    Image<BGR> h_then_v  = img | Flip{Axis::Horizontal} | Flip{Axis::Vertical};

    EXPECT_EQ(cv::norm(both.mat(), h_then_v.mat(), cv::NORM_INF), 0.0);
}
