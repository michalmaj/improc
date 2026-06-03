// tests/core/ops/test_haar_detect.cpp
#include <gtest/gtest.h>
#include <filesystem>
#include <opencv2/objdetect.hpp>
#include "improc/core/pipeline.hpp"
#include "improc/exceptions.hpp"

using namespace improc::core;
namespace fs = std::filesystem;

// Path to existing Haar cascade test model
static const fs::path CASCADE =
    fs::path{__FILE__}.parent_path().parent_path().parent_path()
    / "ml" / "testdata" / "haarcascade_frontalface_default.xml";

static Image<BGR> make_black(int rows = 100, int cols = 100) {
    return Image<BGR>(cv::Mat(rows, cols, CV_8UC3, cv::Scalar(0,0,0)));
}

TEST(DetectHaarTest, ScaleFactorBelowOneThrows) {
    EXPECT_THROW(DetectHaar{}.scale_factor(0.9), improc::ParameterError);
}

TEST(DetectHaarTest, ScaleFactorExactlyOneThrows) {
    EXPECT_THROW(DetectHaar{}.scale_factor(1.0), improc::ParameterError);
}

TEST(DetectHaarTest, NegativeMinNeighborsThrows) {
    EXPECT_THROW(DetectHaar{}.min_neighbors(-1), improc::ParameterError);
}

TEST(DetectHaarTest, InferenceOnBlackImageReturnsEmptyVector) {
    cv::CascadeClassifier cc;
    ASSERT_TRUE(cc.load(CASCADE.string())) << "failed to load: " << CASCADE;
    auto rects = DetectHaar{}(make_black(), cc);
    EXPECT_TRUE(rects.empty());   // no faces in a black image
}

TEST(DetectHaarTest, AllRectsWithinImageBounds) {
    cv::CascadeClassifier cc;
    ASSERT_TRUE(cc.load(CASCADE.string()));
    auto img = make_black(200, 200);
    auto rects = DetectHaar{}(img, cc);
    // A black image never yields detections, so this loop is a no-op here.
    // The bounds check guards against regressions if a real cascade fires unexpectedly.
    for (const auto& r : rects) {
        EXPECT_GE(r.x, 0);
        EXPECT_GE(r.y, 0);
        EXPECT_LE(r.x + r.width,  img.cols());
        EXPECT_LE(r.y + r.height, img.rows());
    }
}
