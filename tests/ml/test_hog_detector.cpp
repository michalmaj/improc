// tests/ml/test_hog_detector.cpp
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include "improc/ml/hog_detector.hpp"
#include "improc/exceptions.hpp"

using namespace improc::ml;

static Image<BGR> make_grey(int rows = 400, int cols = 400) {
    return Image<BGR>(cv::Mat(rows, cols, CV_8UC3, cv::Scalar(128, 128, 128)));
}

TEST(HOGDetectorTest, DefaultConstructorSucceeds) {
    EXPECT_NO_THROW(HOGDetector{});
}

TEST(HOGDetectorTest, EmptySvmThrows) {
    EXPECT_THROW(HOGDetector{std::vector<float>{}}, improc::ParameterError);
}

TEST(HOGDetectorTest, ScaleBelowOneThrows) {
    EXPECT_THROW(HOGDetector{}.scale(0.9), improc::ParameterError);
}

TEST(HOGDetectorTest, ScaleExactlyOneThrows) {
    EXPECT_THROW(HOGDetector{}.scale(1.0), improc::ParameterError);
}

TEST(HOGDetectorTest, InferenceOnGreyImageReturnsVector) {
    // grey 400x400 has no people — must return empty without throwing
    auto result = HOGDetector{}(make_grey());
    EXPECT_TRUE(result.empty());
}

TEST(HOGDetectorTest, ConfidenceInZeroOne) {
    auto result = HOGDetector{}.hit_threshold(-5.0)(make_grey());
    if (result.empty()) GTEST_SKIP() << "no HOG detections on grey image — property untestable";
    for (const auto& det : result) {
        EXPECT_GE(det.confidence, 0.0f);
        EXPECT_LE(det.confidence, 1.0f);
    }
}

TEST(HOGDetectorTest, ClassIdIsZero) {
    auto result = HOGDetector{}.hit_threshold(-5.0)(make_grey());
    if (result.empty()) GTEST_SKIP() << "no HOG detections on grey image — property untestable";
    for (const auto& det : result) {
        EXPECT_EQ(det.class_id, 0);
    }
}

TEST(HOGDetectorTest, LabelIsPerson) {
    auto result = HOGDetector{}.hit_threshold(-5.0)(make_grey());
    if (result.empty()) GTEST_SKIP() << "no HOG detections on grey image — property untestable";
    for (const auto& det : result) {
        EXPECT_EQ(det.label, "person");
    }
}
